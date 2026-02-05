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

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
