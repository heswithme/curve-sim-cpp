// Arbitrage decision logic (floating-point only)
// Currently twocrypto_fx specific - will refactor when adding other pool types
#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "pools/twocrypto_fx/helpers.hpp"
#include "trading/costs.hpp"
#include "trading/decision.hpp"

namespace arb {
namespace trading {

namespace fx = arb::pools::twocrypto_fx;

inline constexpr int NUMERIC_SEARCH_ITERS = 24;

template <typename T, typename PoolT>
Decision<T> decide_trade(
    const PoolT& pool,
    T cex_price,
    const Costs<T>& costs,
    T volume_cap,
    T min_swap_frac,
    T max_swap_frac,
    T cex_fee_discount,
    T cex_fee_markup
) {
    static_assert(std::is_floating_point_v<T>, "decide_trade is floating-only");

    Decision<T> d{};

    if (!(cex_price > T(0))) return d;

    const auto xp_now = fx::pool_xp_current(pool);
    const T p_now = fx::MathOps<T>::get_p(xp_now, pool.D, {pool.A, pool.gamma}) * pool.cached_price_scale;
    const T fee_pool = pool.fee(xp_now);

    const T one_minus_fee = std::max(T(1) - fee_pool, T(1e-12));
    const T p_pool_bid = one_minus_fee * p_now;
    const T p_pool_ask = p_now / one_minus_fee;

    const T p_cex_bid = cex_fee_discount * cex_price;
    const T p_cex_ask = cex_fee_markup * cex_price;

    // Check for arb edge
    const T edge_01 = p_cex_bid - p_pool_ask;  // buy pool, sell CEX
    const T edge_10 = p_pool_bid - p_cex_ask;  // buy CEX, sell pool

    if (edge_01 <= T(0) && edge_10 <= T(0)) return d;
    d.edge_seen = true;
    int sel_i = -1, sel_j = -1;
    if (edge_01 >= edge_10) { sel_i = 0; sel_j = 1; } else { sel_i = 1; sel_j = 0; }

    const T avail = pool.balances[static_cast<size_t>(sel_i)];
    if (!(avail > T(0))) return d;

    // Sizing bounds
    T dx_lo = std::max(T(1e-18), avail * std::max(T(1e-12), min_swap_frac));
    T dx_hi = avail * max_swap_frac;

    if (std::isfinite(static_cast<double>(volume_cap)) && volume_cap > T(0)) {
        T cap = volume_cap;
        if (costs.volume_cap_is_coin1) {
            if (sel_i == 0) {
                cap *= cex_price; // coin1 -> coin0
            }
        } else {
            if (sel_i == 1) {
                cap /= cex_price; // coin0 -> coin1
            }
        }
        dx_hi = std::min(dx_hi, cap);
    }
    if (!(dx_hi > dx_lo)) {
        d.rejected_invalid_size = true;
        return d;
    }

    struct Candidate {
        T dx{};
        T dy_after_fee{};
        T profit{};
        T fee_tokens{};
    };

    auto evaluate_candidate = [&](T dx) -> Candidate {
        auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx);
        const T dy_after_fee = sim.first;
        T profit = (sel_i == 0)
            ? (dy_after_fee * cex_price * cex_fee_discount - dx - costs.gas_coin0)
            : (dy_after_fee - dx * cex_price * cex_fee_markup - costs.gas_coin0);
        return Candidate{dx, dy_after_fee, profit, sim.second};
    };

    // Profit-maximize within sizing bounds.
    constexpr T phi = static_cast<T>(0x1.3c6ef372fe95p-1);  // (sqrt(5) - 1) / 2
    T a = dx_lo;
    T b = dx_hi;
    Candidate c = evaluate_candidate(b - phi * (b - a));
    Candidate e = evaluate_candidate(a + phi * (b - a));
    for (int it = 0; it < NUMERIC_SEARCH_ITERS; ++it) {
        if (c.profit < e.profit) {
            a = c.dx;
            c = e;
            e = evaluate_candidate(a + phi * (b - a));
        } else {
            b = e.dx;
            e = c;
            c = evaluate_candidate(b - phi * (b - a));
        }
    }

    Candidate best = (c.profit > e.profit) ? c : e;

    if (!(best.profit > T(0))) {
        d.rejected_nonpositive_profit = true;
        d.profit = best.profit;
        return d;
    }

    d.do_trade = true;
    d.i = sel_i;
    d.j = sel_j;
    d.dx = best.dx;
    d.dy_after_fee = best.dy_after_fee;
    d.profit = best.profit;
    d.fee_tokens = best.fee_tokens;
    d.notional_coin0 = (sel_i == 0) ? best.dx : best.dx * cex_price;

    return d;
}

} // namespace trading
} // namespace arb
