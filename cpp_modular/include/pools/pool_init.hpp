// Pool initialization value types.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <boost/json.hpp>

#include "pools/twocrypto_fx/policy.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"

namespace arb {
namespace pools {

namespace detail {

template <typename T>
T default_precision_fraction(uint64_t num, uint64_t den) {
    return twocrypto_fx::PoolTraits<T>::PRECISION() * T(num) / T(den);
}

template <typename T>
T default_fee_fraction(uint64_t num, uint64_t den) {
    return twocrypto_fx::PoolTraits<T>::FEE_PRECISION() * T(num) / T(den);
}

template <typename T>
T default_amount(uint64_t whole_units) {
    return twocrypto_fx::PoolTraits<T>::PRECISION() * T(whole_units);
}

} // namespace detail

// Pool initialization parameters.
template <typename PoolT, typename HarnessT = PoolT>
struct PoolInit {
    std::array<PoolT, 2> precisions{PoolT(1), PoolT(1)};
    PoolT A{PoolT(100000)};
    PoolT gamma{PoolT(0)};
    PoolT mid_fee{detail::default_fee_fraction<PoolT>(3, 10000)};
    PoolT out_fee{detail::default_fee_fraction<PoolT>(5, 10000)};
    PoolT fee_gamma{detail::default_precision_fraction<PoolT>(23, 100)};
    PoolT adjustment_step_min{detail::default_precision_fraction<PoolT>(1, 1000000)};
    PoolT adjustment_step_max{detail::default_precision_fraction<PoolT>(1, 1000)};
    PoolT ma_time{PoolT(600)};
    PoolT reserved_profit_fraction{detail::default_fee_fraction<PoolT>(1, 2)};
    PoolT admin_fee{detail::default_fee_fraction<PoolT>(1, 2)};
    twocrypto_fx::PolicyKind policy_kind{twocrypto_fx::PolicyKind::None};
    twocrypto_fx::PolicyConfig<PoolT> policy_config{};
    PoolT initial_price{detail::default_precision_fraction<PoolT>(1, 1)};
    std::array<PoolT, 2> initial_liq{
        detail::default_amount<PoolT>(1000000),
        detail::default_amount<PoolT>(1000000)
    };
    uint64_t start_ts{0};

    HarnessT donation_apy{HarnessT(0)};
    HarnessT donation_frequency{HarnessT(0)};
    PoolT donation_duration{PoolT(7 * 86400)};
    HarnessT donation_coins_ratio{HarnessT(0.5)};

    std::string tag;
    size_t global_index{0};
    boost::json::object echo_pool{};
    boost::json::object echo_costs{};
};

} // namespace pools
} // namespace arb
