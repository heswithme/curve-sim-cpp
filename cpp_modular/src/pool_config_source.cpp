#include "pools/pool_config_source.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/json_utils.hpp"

namespace arb {
namespace pools {
namespace {

const boost::json::object* get_meta_object(const boost::json::object& root) {
    if (auto* v = root.if_contains("meta")) {
        if (v->is_object()) return &v->as_object();
    }
    if (auto* v = root.if_contains("metadata")) {
        if (v->is_object()) return &v->as_object();
    }
    return nullptr;
}

bool is_grid_axis_key(const std::string& key) {
    if (key.size() < 2 || key[0] != 'x') return false;
    for (size_t i = 1; i < key.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(key[i]))) return false;
    }
    return true;
}

bool json_bool_opt(const boost::json::object& obj, const char* key, bool fallback = false) {
    if (auto* v = obj.if_contains(key)) {
        if (v->is_bool()) return v->as_bool();
    }
    return fallback;
}

std::string json_value_to_tag(const boost::json::value& v) {
    if (v.is_string()) return std::string(v.as_string().c_str());
    if (v.is_int64()) return std::to_string(v.as_int64());
    if (v.is_uint64()) return std::to_string(v.as_uint64());
    if (v.is_double()) {
        std::ostringstream oss;
        oss.precision(17);
        oss << v.as_double();
        return oss.str();
    }
    if (v.is_bool()) return v.as_bool() ? "true" : "false";
    return boost::json::serialize(v);
}

void set_dotted_json_value(
    boost::json::object& obj,
    const std::string& dotted_name,
    const boost::json::value& value
) {
    boost::json::object* cur = &obj;
    size_t start = 0;
    while (true) {
        const size_t dot = dotted_name.find('.', start);
        const std::string part = dotted_name.substr(
            start,
            dot == std::string::npos ? dot : dot - start
        );
        if (part.empty()) {
            throw std::runtime_error("empty compact grid path segment in '" + dotted_name + "'");
        }
        if (dot == std::string::npos) {
            (*cur)[part] = value;
            return;
        }
        auto& next = (*cur)[part];
        if (!next.is_object()) {
            next = boost::json::object{};
        }
        cur = &next.as_object();
        start = dot + 1;
    }
}

void apply_grid_value(
    boost::json::object& pool,
    boost::json::object& costs,
    const std::string& name,
    const boost::json::value& value
) {
    constexpr const char* costs_prefix = "costs.";
    constexpr size_t costs_prefix_len = 6;
    if (name == "policy.fee_bps") {
        boost::json::object policy;
        policy["kind"] = "fixed_fee";
        policy["fee_bps"] = value;
        pool["policy"] = std::move(policy);
        return;
    }
    if (name.rfind(costs_prefix, 0) == 0) {
        set_dotted_json_value(costs, name.substr(costs_prefix_len), value);
        return;
    }
    set_dotted_json_value(pool, name, value);
}

std::vector<GridAxis> ordered_grid_axes(const boost::json::object& meta) {
    auto* grid_value = meta.if_contains("grid");
    if (!grid_value || !grid_value->is_object()) {
        throw std::runtime_error("compact pool config requires meta.grid");
    }
    const auto& grid = grid_value->as_object();
    std::vector<GridAxis> axes;
    axes.reserve(grid.size());
    for (const auto& kv : grid) {
        const std::string key(kv.key().data(), kv.key().size());
        if (!is_grid_axis_key(key)) continue;
        if (!kv.value().is_object()) {
            throw std::runtime_error("compact grid axis " + key + " must be an object");
        }
        const auto& axis_obj = kv.value().as_object();
        auto* name_value = axis_obj.if_contains("name");
        auto* values_value = axis_obj.if_contains("values");
        if (!name_value || !name_value->is_string() || !values_value || !values_value->is_array()) {
            throw std::runtime_error("compact grid axis " + key + " requires string name and array values");
        }
        const auto& values = values_value->as_array();
        if (values.empty()) {
            throw std::runtime_error("compact grid axis " + key + " has no values");
        }
        axes.push_back(GridAxis{
            static_cast<size_t>(std::stoull(key.substr(1))),
            std::string(name_value->as_string().c_str()),
            values
        });
    }
    std::sort(axes.begin(), axes.end(), [](const GridAxis& a, const GridAxis& b) {
        return a.ordinal < b.ordinal;
    });
    if (axes.empty()) {
        throw std::runtime_error("compact pool config requires at least one grid axis");
    }
    return axes;
}

size_t checked_grid_count(const std::vector<GridAxis>& axes) {
    size_t count = 1;
    for (const auto& axis : axes) {
        const size_t n = axis.values.size();
        if (count > std::numeric_limits<size_t>::max() / n) {
            throw std::runtime_error("compact grid pool count overflows size_t");
        }
        count *= n;
    }
    return count;
}

boost::json::object make_compact_grid_entry(
    const boost::json::object& base_pool,
    const boost::json::object& base_costs,
    const std::vector<GridAxis>& axes,
    size_t index,
    bool fee_equalize
) {
    boost::json::object pool = base_pool;
    boost::json::object costs = base_costs;
    std::vector<size_t> strides(axes.size(), 1);
    for (size_t i = axes.size(); i-- > 0;) {
        if (i + 1 < axes.size()) {
            strides[i] = strides[i + 1] * axes[i + 1].values.size();
        }
    }

    bool touches_fee = false;
    std::ostringstream tag;
    for (size_t i = 0; i < axes.size(); ++i) {
        const auto& axis = axes[i];
        const size_t coord = (index / strides[i]) % axis.values.size();
        const auto& value = axis.values.at(coord);
        if (
            axis.name != "policy.fee_bps" &&
            axis.name.rfind("costs.", 0) != 0 &&
            axis.name.find('.') == std::string::npos
        ) {
            apply_grid_value(pool, costs, axis.name, boost::json::value(json_value_to_tag(value)));
        } else {
            apply_grid_value(pool, costs, axis.name, value);
        }
        touches_fee = touches_fee || axis.name == "mid_fee" || axis.name == "out_fee";
        if (i > 0) tag << "__";
        tag << axis.name << "_" << json_value_to_tag(value);
    }

    if (fee_equalize && touches_fee) {
        if (auto* mid = pool.if_contains("mid_fee")) {
            pool["out_fee"] = *mid;
        }
    }

    boost::json::object entry;
    entry["tag"] = tag.str();
    entry["pool"] = std::move(pool);
    if (!costs.empty()) {
        entry["costs"] = std::move(costs);
    }
    return entry;
}

} // namespace

PoolConfigDocument::PoolConfigDocument(const std::string& path)
    : root_(boost::json::parse(read_file(path)))
{
    if (root_.is_object()) {
        const auto& obj = root_.as_object();
        if (obj.contains("pools")) {
            if (!obj.at("pools").is_array()) {
                throw std::runtime_error("Invalid pools json: 'pools' must be an array");
            }
            mode_ = Mode::ExpandedPools;
            total_ = obj.at("pools").as_array().size();
            return;
        }
        if (obj.contains("pool")) {
            mode_ = Mode::SinglePool;
            total_ = 1;
            return;
        }

        const auto* meta = get_meta_object(obj);
        if (!meta) {
            throw std::runtime_error("Invalid pools json: expected 'pools' array, single 'pool', or compact meta.base_pool/meta.grid");
        }
        auto* base_pool_value = meta->if_contains("base_pool");
        if (!base_pool_value || !base_pool_value->is_object()) {
            throw std::runtime_error("compact pool config requires meta.base_pool");
        }
        base_pool_ = base_pool_value->as_object();

        if (auto* base_costs_value = meta->if_contains("base_costs")) {
            if (!base_costs_value->is_object()) {
                throw std::runtime_error("compact pool config meta.base_costs must be an object");
            }
            base_costs_ = base_costs_value->as_object();
        }
        axes_ = ordered_grid_axes(*meta);
        total_ = checked_grid_count(axes_);
        fee_equalize_ = json_bool_opt(*meta, "fee_equalize", false);
        mode_ = Mode::CompactGrid;
        return;
    }

    if (root_.is_array()) {
        mode_ = Mode::ExpandedArray;
        total_ = root_.as_array().size();
        return;
    }

    throw std::runtime_error("Invalid pools json root type");
}

size_t PoolConfigDocument::size() const {
    return total_;
}

boost::json::object PoolConfigDocument::entry_at(size_t index) const {
    if (index >= total_) {
        throw std::out_of_range("pool config index out of range");
    }

    switch (mode_) {
    case Mode::ExpandedPools:
        return root_.as_object().at("pools").as_array().at(index).as_object();
    case Mode::SinglePool:
        return root_.as_object();
    case Mode::ExpandedArray:
        return root_.as_array().at(index).as_object();
    case Mode::CompactGrid:
        return make_compact_grid_entry(base_pool_, base_costs_, axes_, index, fee_equalize_);
    }

    throw std::runtime_error("unknown pool config source mode");
}

} // namespace pools
} // namespace arb
