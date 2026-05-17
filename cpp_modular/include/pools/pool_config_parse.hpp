// Templated pool config parsing.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/json.hpp>

#include "core/json_utils.hpp"
#include "harness/pool_value.hpp"
#include "pools/pool_init.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"
#include "trading/costs.hpp"

namespace arb {
namespace pools {

inline bool is_number_or_string(const boost::json::value& v) {
    return v.is_string() || v.is_double() || v.is_int64() || v.is_uint64();
}

inline std::string scalar_to_string(const boost::json::value& v) {
    if (v.is_string()) return std::string(v.as_string().c_str());
    if (v.is_int64()) return std::to_string(v.as_int64());
    if (v.is_uint64()) return std::to_string(v.as_uint64());
    if (v.is_double()) {
        std::ostringstream oss;
        oss << std::setprecision(17) << v.as_double();
        return oss.str();
    }
    return "0";
}

inline std::string integer_string_from_scalar(const boost::json::value& v) {
    const std::string s = scalar_to_string(v);
    if (s.find_first_of(".eE") == std::string::npos) {
        return s;
    }
    const long double x = std::strtold(s.c_str(), nullptr);
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(0);
    oss << std::floor(x + 0.5L);
    return oss.str();
}

template <typename T>
T parse_pool_plain(const boost::json::value& v) {
    if constexpr (harness::is_uint_pool_v<T>) {
        return T(integer_string_from_scalar(v));
    } else {
        return parse_plain_real<T>(v);
    }
}

template <typename T>
T parse_pool_wad(const boost::json::value& v) {
    if constexpr (harness::is_uint_pool_v<T>) {
        return T(integer_string_from_scalar(v));
    } else {
        return parse_scaled_1e18<T>(v);
    }
}

template <typename T>
T parse_pool_fee(const boost::json::value& v) {
    if constexpr (harness::is_uint_pool_v<T>) {
        return T(integer_string_from_scalar(v));
    } else {
        return parse_fee_1e10<T>(v);
    }
}

template <typename PoolT, typename HarnessT = PoolT>
twocrypto_fx::PolicyConfig<PoolT> parse_policy_config(const boost::json::value& policy) {
    twocrypto_fx::PolicyConfig<PoolT> cfg{};
    if (policy.is_string()) {
        cfg.kind = twocrypto_fx::policy_kind_from_string(std::string(policy.as_string().c_str()));
        return cfg;
    }
    if (!policy.is_object()) {
        throw std::runtime_error("pool policy must be a string or object");
    }

    const auto& po = policy.as_object();
    std::string kind = "none";
    if (auto* k = po.if_contains("kind")) {
        if (!k->is_string()) {
            throw std::runtime_error("pool policy kind must be a string");
        }
        kind = std::string(k->as_string().c_str());
    }
    cfg.kind = twocrypto_fx::policy_kind_from_string(kind);

    if (auto* fee = po.if_contains("fee")) {
        if (!is_number_or_string(*fee)) {
            throw std::runtime_error("pool policy fee must be a string or number");
        }
        cfg.fee = parse_pool_fee<PoolT>(*fee);
    } else if (auto* fee_bps = po.if_contains("fee_bps")) {
        if (!is_number_or_string(*fee_bps)) {
            throw std::runtime_error("pool policy fee_bps must be a string or number");
        }
        const HarnessT bps = parse_plain_real<HarnessT>(*fee_bps);
        cfg.fee = harness::h_fee_to_pool<PoolT>(bps / HarnessT(10000));
    }
    return cfg;
}

// Entry format: { "pool": {...}, "costs": {...}, "tag": "..." }
// Or just pool params directly: { "A": ..., "gamma": ..., ... }.
template <typename PoolT, typename HarnessT = PoolT>
void parse_pool_entry(
    const boost::json::object& entry,
    PoolInit<PoolT, HarnessT>& out_pool,
    arb::trading::Costs<HarnessT>& out_costs
) {
    namespace json = boost::json;

    const json::object& pool = entry.contains("pool")
        ? entry.at("pool").as_object()
        : entry;

    out_pool.echo_pool = pool;

    if (auto* v = entry.if_contains("tag")) {
        if (v->is_string()) out_pool.tag = v->as_string().c_str();
    }

    if (auto* v = pool.if_contains("initial_liquidity")) {
        const auto& a = v->as_array();
        if (a.size() >= 2) {
            out_pool.initial_liq[0] = parse_pool_wad<PoolT>(a[0]);
            out_pool.initial_liq[1] = parse_pool_wad<PoolT>(a[1]);
        }
    }

    if (auto* v = pool.if_contains("A")) out_pool.A = parse_pool_plain<PoolT>(*v);
    if (auto* v = pool.if_contains("gamma")) out_pool.gamma = parse_pool_plain<PoolT>(*v);
    if (auto* v = pool.if_contains("mid_fee")) out_pool.mid_fee = parse_pool_fee<PoolT>(*v);
    if (auto* v = pool.if_contains("out_fee")) out_pool.out_fee = parse_pool_fee<PoolT>(*v);
    if (auto* v = pool.if_contains("fee_gamma")) out_pool.fee_gamma = parse_pool_wad<PoolT>(*v);
    if (
        pool.if_contains("lp_profit_fraction") ||
        pool.if_contains("allowed_extra_profit") ||
        pool.if_contains("adjustment_step") ||
        pool.if_contains("fee_params") ||
        pool.if_contains("fee_model_name")
    ) {
        throw std::runtime_error("legacy pool config fields are not supported; use reserved_profit_fraction, admin_fee, adjustment_step_min, adjustment_step_max, and policy");
    }
    if (auto* v = pool.if_contains("adjustment_step_min")) out_pool.adjustment_step_min = parse_pool_wad<PoolT>(*v);
    if (auto* v = pool.if_contains("adjustment_step_max")) out_pool.adjustment_step_max = parse_pool_wad<PoolT>(*v);
    if (auto* v = pool.if_contains("ma_time")) out_pool.ma_time = parse_pool_plain<PoolT>(*v);
    if (auto* v = pool.if_contains("reserved_profit_fraction")) {
        out_pool.reserved_profit_fraction = std::clamp<PoolT>(
            parse_pool_fee<PoolT>(*v),
            PoolT(0),
            twocrypto_fx::PoolTraits<PoolT>::FEE_PRECISION()
        );
    }
    if (auto* v = pool.if_contains("admin_fee")) {
        out_pool.admin_fee = std::clamp<PoolT>(
            parse_pool_fee<PoolT>(*v),
            PoolT(0),
            twocrypto_fx::PoolTraits<PoolT>::FEE_PRECISION()
        );
    }
    if (auto* v = pool.if_contains("policy")) {
        out_pool.policy_config = parse_policy_config<PoolT, HarnessT>(*v);
        out_pool.policy_kind = out_pool.policy_config.kind;
    }
    if (auto* v = pool.if_contains("initial_price")) out_pool.initial_price = parse_pool_wad<PoolT>(*v);
    if (auto* v = pool.if_contains("start_timestamp")) {
        out_pool.start_ts = static_cast<uint64_t>(parse_plain_real<HarnessT>(*v));
        if (out_pool.start_ts > 10000000000ULL) {
            out_pool.start_ts /= 1000ULL;
        }
    }

    if (auto* v = pool.if_contains("donation_apy")) out_pool.donation_apy = parse_plain_real<HarnessT>(*v);
    if (auto* v = pool.if_contains("donation_frequency")) out_pool.donation_frequency = parse_plain_real<HarnessT>(*v);
    if (auto* v = pool.if_contains("donation_duration")) out_pool.donation_duration = parse_pool_plain<PoolT>(*v);
    if (auto* v = pool.if_contains("donation_coins_ratio")) {
        HarnessT r = parse_plain_real<HarnessT>(*v);
        out_pool.donation_coins_ratio = std::clamp<HarnessT>(r, HarnessT(0), HarnessT(1));
    }

    if (auto* c = entry.if_contains("costs")) {
        const auto& co = c->as_object();
        out_pool.echo_costs = co;
        if (auto* v = co.if_contains("arb_fee_bps")) out_costs.arb_fee_bps = parse_plain_real<HarnessT>(*v);
        if (auto* v = co.if_contains("gas_coin0")) out_costs.gas_coin0 = parse_plain_real<HarnessT>(*v);
        if (auto* v = co.if_contains("use_volume_cap")) out_costs.use_volume_cap = v->as_bool();
        if (auto* v = co.if_contains("volume_cap_mult")) out_costs.volume_cap_mult = parse_plain_real<HarnessT>(*v);
        if (auto* v = co.if_contains("volume_cap_is_coin_1")) {
            out_costs.volume_cap_is_coin1 = v->is_bool()
                ? v->as_bool()
                : (parse_plain_real<HarnessT>(*v) != HarnessT(0));
        }
    }
}

} // namespace pools
} // namespace arb
