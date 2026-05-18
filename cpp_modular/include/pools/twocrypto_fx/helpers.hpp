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

    const T balance0 = pool.balances[0] + (i == 0 ? dx : T(0));
    const T balance1 = pool.balances[1] + (i == 1 ? dx : T(0));
    std::array<T, 2> xp{
        balance0 * pool.precisions[0],
        balance1 * pool.precisions[1] * ps / PoolTraits<T>::PRECISION()
    };

    auto y_out = MathOps<T>::get_y_unchecked(pool.A, pool.gamma, xp, pool.D, j);
    T dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    T dy_tokens  = xp_to_tokens_j(pool, j, dy_xp, ps);
    T fee_pool   = pool.fee(xp);
    T fee_tokens = fee_pool * dy_tokens / PoolTraits<T>::FEE_PRECISION();
    return {dy_tokens - fee_tokens, fee_tokens};
}

template <typename T>
inline T quoted_exchange_fee_fraction(
    const PoolT<T>& pool,
    size_t i,
    size_t j,
    T dx
) {
    if (!(dx > T(0))) {
        return pool.fee(pool_xp_current(pool));
    }

    const T ps = pool.cached_price_scale;

    const T balance0 = pool.balances[0] + (i == 0 ? dx : T(0));
    const T balance1 = pool.balances[1] + (i == 1 ? dx : T(0));
    std::array<T, 2> xp{
        balance0 * pool.precisions[0],
        balance1 * pool.precisions[1] * ps / PoolTraits<T>::PRECISION()
    };

    auto y_out = MathOps<T>::get_y_unchecked(pool.A, pool.gamma, xp, pool.D, j);
    T dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    return pool.fee(xp);
}

template <typename T>
inline T viewer_exchange_fee_fraction(const PoolT<T>& pool, T cex_price) {
    const T min_dx0 = pool.balances[0] / T(1000000);
    const T min_dx1 = pool.balances[1] / T(1000000);

    if (cex_price > T(0)) {
        const auto xp_now = pool_xp_current(pool);
        const T p_pool = MathOps<T>::get_p(
            xp_now,
            pool.D,
            {pool.A, pool.gamma}
        ) * pool.cached_price_scale;
        if (cex_price >= p_pool && min_dx0 > T(0)) {
            return quoted_exchange_fee_fraction(pool, 0, 1, min_dx0);
        }
        if (min_dx1 > T(0)) {
            return quoted_exchange_fee_fraction(pool, 1, 0, min_dx1);
        }
    }

    T sum = T(0);
    T count = T(0);
    if (min_dx0 > T(0)) {
        sum += quoted_exchange_fee_fraction(pool, 0, 1, min_dx0);
        count += T(1);
    }
    if (min_dx1 > T(0)) {
        sum += quoted_exchange_fee_fraction(pool, 1, 0, min_dx1);
        count += T(1);
    }
    if (count > T(0)) {
        return sum / count;
    }
    return pool.fee(pool_xp_current(pool));
}

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
