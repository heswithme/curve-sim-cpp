// Pool initialization value types.
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <boost/json.hpp>

#include "pools/twocrypto_fx/policy.hpp"

namespace arb {
namespace pools {

// Pool initialization parameters (floating-point, unit-scaled).
template <typename T>
struct PoolInit {
    std::array<T, 2> precisions{T(1), T(1)};
    T A{T(100000.0)};
    T gamma{T(0)};
    T mid_fee{T(3e-4)};
    T out_fee{T(5e-4)};
    T fee_gamma{T(0.23)};
    T adjustment_step_min{T(1e-6)};
    T adjustment_step_max{T(1e-3)};
    T ma_time{T(600.0)};
    T reserved_profit_fraction{T(0.5)};
    T admin_fee{T(0.5)};
    twocrypto_fx::PolicyKind policy_kind{twocrypto_fx::PolicyKind::None};
    twocrypto_fx::PolicyConfig<T> policy_config{};
    T initial_price{T(1.0)};
    std::array<T, 2> initial_liq{T(1e6), T(1e6)};
    uint64_t start_ts{0};

    T donation_apy{T(0)};
    T donation_frequency{T(0)};
    T donation_duration{T(7 * 86400)};
    T donation_coins_ratio{T(0.5)};

    std::string tag;
    boost::json::object echo_pool{};
    boost::json::object echo_costs{};
};

} // namespace pools
} // namespace arb
