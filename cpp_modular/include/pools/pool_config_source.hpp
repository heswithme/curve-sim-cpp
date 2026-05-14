// Pool config source facade: expanded JSON entries or compact generated grids.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <boost/json.hpp>

#include "pools/pool_config_parse.hpp"
#include "trading/costs.hpp"

namespace arb {
namespace pools {

struct GridAxis {
    size_t ordinal{0};
    std::string name;
    boost::json::array values{};
};

class PoolConfigDocument {
public:
    explicit PoolConfigDocument(const std::string& path);

    size_t size() const;
    boost::json::object entry_at(size_t index) const;

private:
    enum class Mode {
        ExpandedPools,
        SinglePool,
        ExpandedArray,
        CompactGrid,
    };

    boost::json::value root_;
    Mode mode_{Mode::ExpandedArray};
    size_t total_{0};

    boost::json::object base_pool_{};
    boost::json::object base_costs_{};
    std::vector<GridAxis> axes_{};
    bool fee_equalize_{false};
};

// Load pool entries from a JSON file.
// Supports:
// - { "pools": [ {...}, {...} ] }
// - { "pool": {...} }
// - [ {...}, {...} ]
// - { "meta": { "base_pool": {...}, "base_costs": {...}, "grid": {...} } }
// Optional: start_idx/end_idx for loading a subset (0-based, end exclusive).
template <typename T>
std::vector<std::pair<PoolInit<T>, arb::trading::Costs<T>>>
load_pool_configs(const std::string& path, size_t start_idx = 0, size_t end_idx = SIZE_MAX) {
    PoolConfigDocument doc(path);
    if (start_idx >= doc.size()) {
        return {};
    }
    const size_t actual_end = std::min(end_idx, doc.size());

    std::vector<std::pair<PoolInit<T>, arb::trading::Costs<T>>> result;
    result.reserve(actual_end - start_idx);

    for (size_t i = start_idx; i < actual_end; ++i) {
        auto entry = doc.entry_at(i);
        PoolInit<T> pool_init{};
        arb::trading::Costs<T> costs{};
        parse_pool_entry(entry, pool_init, costs);
        result.emplace_back(std::move(pool_init), std::move(costs));
    }

    return result;
}

} // namespace pools
} // namespace arb
