// Helper utilities for twocrypto_fx pool
#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <utility>

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

template <typename T>
inline FeeBreakdown<T> state_fee_breakdown(
    const PoolT<T>& pool,
    const std::array<T, 2>& xp
) {
    return pool.state_fee_breakdown(xp);
}

template <typename T>
inline std::pair<T, T> post_trade_price_and_fee(
    const PoolT<T>& pool,
    size_t i,
    size_t j,
    T dx,
    T cex_price = T(0)
) {
    const auto preview = pool.preview_exchange(i, j, dx, cex_price);
    return {preview.spot_post, preview.fees.total_fee};
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
    T dx,
    T cex_price = T(0)
) {
    const auto preview = pool.preview_exchange(i, j, dx, cex_price);
    return {preview.dy_after_fee, preview.fee_tokens};
}

// -----------------------------------------------------------------------------
// Convenience conversion
// -----------------------------------------------------------------------------
template <typename T>
inline T coin0_equiv(const T& amt0, const T& amt1, const T& price_scale) {
    return amt0 + amt1 * price_scale / PoolTraits<T>::PRECISION();
}

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
