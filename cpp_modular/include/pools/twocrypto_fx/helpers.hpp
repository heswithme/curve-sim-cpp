// Helper utilities for twocrypto_fx pool
#pragma once

#include <array>
#include <boost/json.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "core/json_utils.hpp"
#include "pools/twocrypto_fx/stableswap_math.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"

namespace arb {
namespace pools {
namespace twocrypto_fx {

template <typename T>
using PoolT = TwoCryptoPool<T>;

// -----------------------------------------------------------------------------
// Balance/vector helpers
// -----------------------------------------------------------------------------
template <typename T>
inline std::array<T, 2> pool_xp_from(
    const PoolT<T>& pool,
    const std::array<T, 2>& balances,
    const T& price_scale
) {
    return {
        balances[0] * pool.precisions[0],
        balances[1] * pool.precisions[1] * price_scale / PoolTraits<T>::PRECISION()
    };
}

template <typename T>
inline std::array<T, 2> pool_xp_current(const PoolT<T>& pool) {
    return pool_xp_from(pool, pool.balances, pool.cached_price_scale);
}

template <typename T>
inline T xp_to_tokens_j(const PoolT<T>& pool, size_t j, T amount_xp, const T& price_scale) {
    T v = amount_xp;
    if (j == 1) {
        v = v * PoolTraits<T>::PRECISION() / price_scale;
    }
    return v / pool.precisions[j];
}

// -----------------------------------------------------------------------------
// Fee + price helpers
// -----------------------------------------------------------------------------
template <typename T>
inline T dyn_fee(
    const std::array<T, 2>& xp,
    const T& mid_fee,
    const T& out_fee,
    const T& fee_gamma
) {
    const T Bsum = xp[0] + xp[1];
    if (!(Bsum > T(0))) {
        return mid_fee;
    }

    // Matches the pool's internal _fee formula; for floating types PRECISION() == 1.
    T B = PoolTraits<T>::PRECISION() * PoolT<T>::N_COINS * PoolT<T>::N_COINS * xp[0] / Bsum * xp[1] / Bsum;
    B = fee_gamma * B /
        (fee_gamma * B / PoolTraits<T>::PRECISION() + PoolTraits<T>::PRECISION() - B);

    return (
        mid_fee * B + out_fee * (PoolTraits<T>::PRECISION() - B)
    ) / PoolTraits<T>::PRECISION();
}

template <typename T>
inline std::pair<T, T> post_trade_price_and_fee(
    const PoolT<T>& pool,
    size_t i,
    size_t j,
    T dx
) {
    const T ps = pool.cached_price_scale;

    auto balances_local = pool.balances;
    balances_local[i] += dx;
    auto xp = pool_xp_from(pool, balances_local, ps);

    auto y_out = MathOps<T>::get_y(pool.A, pool.gamma, xp, pool.D, j);
    T dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    const T fee_pool = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    const T D_new = MathOps<T>::newton_D(pool.A, pool.gamma, xp, T(0));
    const T p_new = MathOps<T>::get_p(xp, D_new, {pool.A, pool.gamma}) * ps / PoolTraits<T>::PRECISION();

    return {p_new, fee_pool};
}

template <typename T>
inline T balance_indicator(const PoolT<T>& pool) {
    const T ps = pool.cached_price_scale;
    const auto xp = pool_xp_from(pool, pool.balances, ps);
    const T denom = xp[0] + xp[1];
    if (!(denom > T(0))) {
        return T(0);
    }
    return static_cast<T>(4) * xp[0] * xp[1] / (denom * denom);
}

// -----------------------------------------------------------------------------
// Simulation helper (no state change)
// -----------------------------------------------------------------------------
template <typename T>
inline std::pair<T, T> simulate_exchange_once(
    const PoolT<T>& pool,
    size_t i,
    size_t j,
    T dx
) {
    const T ps = pool.cached_price_scale;

    auto balances_local = pool.balances;
    balances_local[i] += dx;
    auto xp = pool_xp_from(pool, balances_local, ps);

    auto y_out = MathOps<T>::get_y(pool.A, pool.gamma, xp, pool.D, j);
    T dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    T dy_tokens  = xp_to_tokens_j(pool, j, dy_xp, ps);
    T fee_pool   = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    T fee_tokens = fee_pool * dy_tokens / PoolTraits<T>::FEE_PRECISION();
    return {dy_tokens - fee_tokens, fee_tokens};
}

// -----------------------------------------------------------------------------
// Convenience conversion
// -----------------------------------------------------------------------------
template <typename T>
inline T coin0_equiv(const T& amt0, const T& amt1, const T& price_scale) {
    return amt0 + amt1 * price_scale / PoolTraits<T>::PRECISION();
}

// -----------------------------------------------------------------------------
// Runtime view + final-state helpers
// -----------------------------------------------------------------------------
template <typename T>
inline double runtime_plain_to_double(const T& value) {
    return static_cast<double>(value);
}

template <>
inline double runtime_plain_to_double<uint256>(const uint256& value) {
    return static_cast<double>(value.convert_to<long double>());
}

template <typename T>
inline double runtime_wad_to_double(const T& value) {
    return static_cast<double>(value);
}

template <>
inline double runtime_wad_to_double<uint256>(const uint256& value) {
    return static_cast<double>(value.convert_to<long double>() / WAD);
}

template <typename T>
inline double runtime_fee_to_double(const T& value) {
    return static_cast<double>(value);
}

template <>
inline double runtime_fee_to_double<uint256>(const uint256& value) {
    return static_cast<double>(value.convert_to<long double>() / FEE_SCALE);
}

template <typename T>
inline TwoCryptoPool<double> make_runtime_view(const PoolT<T>& src) {
    TwoCryptoPool<double> view(
        {
            runtime_plain_to_double(src.precisions[0]),
            runtime_plain_to_double(src.precisions[1]),
        },
        runtime_plain_to_double(src.A),
        runtime_wad_to_double(src.gamma),
        runtime_fee_to_double(src.mid_fee),
        runtime_fee_to_double(src.out_fee),
        runtime_wad_to_double(src.fee_gamma),
        runtime_wad_to_double(src.allowed_extra_profit),
        runtime_wad_to_double(src.adjustment_step),
        runtime_plain_to_double(src.ma_time),
        runtime_wad_to_double(src.cached_price_scale)
    );

    view.balances = {
        runtime_wad_to_double(src.balances[0]),
        runtime_wad_to_double(src.balances[1]),
    };
    view.D = runtime_wad_to_double(src.D);
    view.totalSupply = runtime_wad_to_double(src.totalSupply);
    view.cached_price_scale = runtime_wad_to_double(src.cached_price_scale);
    view.cached_price_oracle = runtime_wad_to_double(src.cached_price_oracle);
    view.last_prices = runtime_wad_to_double(src.last_prices);
    view.last_timestamp = src.last_timestamp;
    view.A = runtime_plain_to_double(src.A);
    view.gamma = runtime_wad_to_double(src.gamma);
    view.mid_fee = runtime_fee_to_double(src.mid_fee);
    view.out_fee = runtime_fee_to_double(src.out_fee);
    view.fee_gamma = runtime_wad_to_double(src.fee_gamma);
    view.allowed_extra_profit = runtime_wad_to_double(src.allowed_extra_profit);
    view.adjustment_step = runtime_wad_to_double(src.adjustment_step);
    view.ma_time = runtime_plain_to_double(src.ma_time);
    view.xcp_profit = runtime_wad_to_double(src.xcp_profit);
    view.xcp_profit_a = runtime_wad_to_double(src.xcp_profit_a);
    view.virtual_price = runtime_wad_to_double(src.virtual_price);
    view.precisions = {
        runtime_plain_to_double(src.precisions[0]),
        runtime_plain_to_double(src.precisions[1]),
    };
    view.block_timestamp = src.block_timestamp;
    view.donation_shares = runtime_wad_to_double(src.donation_shares);
    view.donation_shares_max_ratio = runtime_wad_to_double(src.donation_shares_max_ratio);
    view.donation_duration = runtime_plain_to_double(src.donation_duration);
    view.last_donation_release_ts = runtime_plain_to_double(src.last_donation_release_ts);
    view.donation_protection_expiry_ts = runtime_plain_to_double(src.donation_protection_expiry_ts);
    view.donation_protection_period = runtime_plain_to_double(src.donation_protection_period);
    view.donation_protection_lp_threshold = runtime_wad_to_double(src.donation_protection_lp_threshold);
    view.admin_fee = runtime_fee_to_double(src.admin_fee);
    view.last_admin_fee_claim_timestamp = src.last_admin_fee_claim_timestamp;
    return view;
}

inline boost::json::object approx_final_state_json(const PoolT<double>& pool) {
    boost::json::object o;
    o["balances"] = boost::json::array{
        to_str_1e18(pool.balances[0]),
        to_str_1e18(pool.balances[1]),
    };
    o["xp"] = boost::json::array{
        to_str_1e18(pool.balances[0] * pool.precisions[0]),
        to_str_1e18(pool.balances[1] * pool.precisions[1] * pool.cached_price_scale),
    };
    o["D"] = to_str_1e18(pool.D);
    o["virtual_price"] = to_str_1e18(pool.get_virtual_price());
    o["xcp_profit"] = to_str_1e18(pool.xcp_profit);
    o["price_scale"] = to_str_1e18(pool.cached_price_scale);
    o["price_oracle"] = to_str_1e18(pool.cached_price_oracle);
    o["last_prices"] = to_str_1e18(pool.last_prices);
    o["totalSupply"] = to_str_1e18(pool.totalSupply);
    o["donation_shares"] = to_str_1e18(pool.donation_shares);
    o["donation_unlocked"] = to_str_1e18(pool.donation_unlocked());
    o["timestamp"] = pool.block_timestamp;
    return o;
}

template <typename T>
inline boost::json::object approx_final_state_json(const PoolT<T>& pool) {
    return approx_final_state_json(make_runtime_view(pool));
}

inline boost::json::object exact_final_state_json(const PoolT<uint256>& pool) {
    boost::json::object o;
    const auto xp = pool_xp_current(pool);
    o["balances"] = boost::json::array{
        pool.balances[0].convert_to<std::string>(),
        pool.balances[1].convert_to<std::string>(),
    };
    o["xp"] = boost::json::array{
        xp[0].convert_to<std::string>(),
        xp[1].convert_to<std::string>(),
    };
    o["D"] = pool.D.convert_to<std::string>();
    o["virtual_price"] = pool.get_virtual_price().convert_to<std::string>();
    o["xcp_profit"] = pool.xcp_profit.convert_to<std::string>();
    o["price_scale"] = pool.cached_price_scale.convert_to<std::string>();
    o["price_oracle"] = pool.cached_price_oracle.convert_to<std::string>();
    o["last_prices"] = pool.last_prices.convert_to<std::string>();
    o["totalSupply"] = pool.totalSupply.convert_to<std::string>();
    o["donation_shares"] = pool.donation_shares.convert_to<std::string>();
    o["donation_unlocked"] = pool.donation_unlocked().convert_to<std::string>();
    o["timestamp"] = pool.block_timestamp;
    return o;
}

template <typename T>
inline boost::json::object runtime_final_state_json(const PoolT<T>& pool) {
    if constexpr (std::is_same_v<T, uint256>) {
        return exact_final_state_json(pool);
    } else {
        return approx_final_state_json(pool);
    }
}

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
