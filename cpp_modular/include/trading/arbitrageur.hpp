// Arbitrage decision logic (floating-point only)
// Currently twocrypto_fx specific - will refactor when adding other pool types
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <boost/math/tools/roots.hpp>

#include "pools/twocrypto_fx/helpers.hpp"
#include "trading/costs.hpp"
#include "trading/decision.hpp"

namespace arb {
namespace trading {

namespace fx = arb::pools::twocrypto_fx;

// Root finder wrapper
template <typename F>
inline bool toms748_root(
    F&& f,
    double lo, double hi,
    double Flo, double Fhi,
    double& out_root,
    unsigned max_iters = 100
) {
    if (!(hi > lo) || !(Flo * Fhi < 0.0)) return false;
    auto tol = boost::math::tools::eps_tolerance<double>(std::numeric_limits<double>::digits10 - 3);
    boost::uintmax_t it = max_iters;
    auto r = boost::math::tools::toms748_solve(std::forward<F>(f), lo, hi, Flo, Fhi, tol, it);
    out_root = (r.first + r.second) / 2.0;
    return true;
}

template <typename T, typename PoolT>
Decision<T> decide_trade(
    const PoolT& pool,
    T cex_price,
    const Costs<T>& costs,
    T volume_cap,
    T min_swap_frac,
    T max_swap_frac
) {
    static_assert(std::is_floating_point_v<T>, "decide_trade is floating-only");

    Decision<T> d{};

    if (!(cex_price > T(0))) return d;

    const T fee_cex = costs.arb_fee_bps / T(10000);

    const auto xp_now = fx::pool_xp_current(pool);
    const T p_now = fx::MathOps<T>::get_p(xp_now, pool.D, {pool.A, pool.gamma}) * pool.cached_price_scale;
    const T fee_pool = fx::dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);

    const T one_minus_fee = std::max(T(1) - fee_pool, T(1e-12));
    const T p_pool_bid = one_minus_fee * p_now;
    const T p_pool_ask = p_now / one_minus_fee;

    const T p_cex_bid = (T(1) - fee_cex) * cex_price;
    const T p_cex_ask = (T(1) + fee_cex) * cex_price;

    // Check for arb edge
    const T edge_01 = p_cex_bid - p_pool_ask;  // buy pool, sell CEX
    const T edge_10 = p_pool_bid - p_cex_ask;  // buy CEX, sell pool

    int sel_i = -1, sel_j = -1;
    if (edge_01 <= T(0) && edge_10 <= T(0)) return d;
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
    if (!(dx_hi > dx_lo)) return d;

    const T dx_range = dx_hi - dx_lo;
    auto profit_at = [&](T dx) -> T {
        auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx);
        const T dy_after_fee = sim.first;
        if (sel_i == 0) {
            return dy_after_fee * cex_price * (T(1) - fee_cex) - dx - costs.gas_coin0;
        }
        return dy_after_fee - dx * cex_price * (T(1) + fee_cex) - costs.gas_coin0;
    };

    // Residual: derivative of profit w.r.t. dx (finite difference)
    auto residual = [&](double dx_d) -> double {
        T dx = static_cast<T>(dx_d);
        T eps = std::max(dx * T(1e-6), dx_range * T(1e-8));
        eps = std::max(eps, T(1e-18));
        T x0 = std::max(dx_lo, dx - eps);
        T x1 = std::min(dx_hi, dx + eps);
        if (!(x1 > x0)) {
            return 0.0;
        }
        T p0 = profit_at(x0);
        T p1 = profit_at(x1);
        return static_cast<double>((p1 - p0) / (x1 - x0));
    };

    double F_lo = residual(static_cast<double>(dx_lo));
    double F_hi = residual(static_cast<double>(dx_hi));

    T dx_star = dx_hi;
    if (F_lo * F_hi < 0.0) {
        double root;
        if (toms748_root(residual, static_cast<double>(dx_lo), static_cast<double>(dx_hi), F_lo, F_hi, root)) {
            dx_star = std::max(static_cast<T>(root), dx_lo);
        }
    } else {
        if (F_lo <= 0.0 && F_hi <= 0.0) {
            return d;
        }
        if (F_lo == 0.0) {
            dx_star = dx_lo;
        } else if (F_hi == 0.0) {
            dx_star = dx_hi;
        } else {
            dx_star = dx_hi;
        }
    }

    // Simulate and compute profit
    auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx_star);
    T dy_after_fee = sim.first;

    T profit;
    if (sel_i == 0) {
        profit = dy_after_fee * cex_price * (T(1) - fee_cex) - dx_star - costs.gas_coin0;
    } else {
        profit = dy_after_fee - dx_star * cex_price * (T(1) + fee_cex) - costs.gas_coin0;
    }

    if (!(profit > T(0))) return d;

    d.do_trade = true;
    d.i = sel_i;
    d.j = sel_j;
    d.dx = dx_star;
    d.profit = profit;
    d.fee_tokens = sim.second;
    d.notional_coin0 = (sel_i == 0) ? dx_star : dx_star * cex_price;

    return d;
}

template <typename T, typename PoolT>
Decision<T> decide_trade_numeric(
    const PoolT& pool,
    T cex_price,
    const Costs<T>& costs,
    T volume_cap,
    T min_swap_frac,
    T max_swap_frac
) {
    static_assert(std::is_floating_point_v<T>, "decide_trade is floating-only");

    Decision<T> d{};

    if (!(cex_price > T(0))) return d;

    const T fee_cex = costs.arb_fee_bps / T(10000);

    const auto xp_now = fx::pool_xp_current(pool);
    const T p_now = fx::MathOps<T>::get_p(xp_now, pool.D, {pool.A, pool.gamma}) * pool.cached_price_scale;
    const T fee_pool = fx::dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);

    const T one_minus_fee = std::max(T(1) - fee_pool, T(1e-12));
    const T p_pool_bid = one_minus_fee * p_now;
    const T p_pool_ask = p_now / one_minus_fee;

    const T p_cex_bid = (T(1) - fee_cex) * cex_price;
    const T p_cex_ask = (T(1) + fee_cex) * cex_price;

    // Check for arb edge
    const T edge_01 = p_cex_bid - p_pool_ask;  // buy pool, sell CEX
    const T edge_10 = p_pool_bid - p_cex_ask;  // buy CEX, sell pool

    int sel_i = -1, sel_j = -1;
    if (edge_01 <= T(0) && edge_10 <= T(0)) return d;
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
    if (!(dx_hi > dx_lo)) return d;

    auto profit_for_dx = [&](T dx) -> T {
        auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx);
        const T dy_after_fee = sim.first;
        return (sel_i == 0)
            ? (dy_after_fee * cex_price * (T(1) - fee_cex) - dx - costs.gas_coin0)
            : (dy_after_fee - dx * cex_price * (T(1) + fee_cex) - costs.gas_coin0);
    };

    // Profit-maximize within sizing bounds (legacy-like numeric sizing)
    const T phi = (std::sqrt(T(5)) - T(1)) / T(2);  // 0.618...
    T a = dx_lo;
    T b = dx_hi;
    T c = b - phi * (b - a);
    T e = a + phi * (b - a);
    T fc = profit_for_dx(c);
    T fe = profit_for_dx(e);
    for (int it = 0; it < 24; ++it) {
        if (fc < fe) {
            a = c;
            c = e;
            fc = fe;
            e = a + phi * (b - a);
            fe = profit_for_dx(e);
        } else {
            b = e;
            e = c;
            fe = fc;
            c = b - phi * (b - a);
            fc = profit_for_dx(c);
        }
    }

    T dx_best = (fc > fe) ? c : e;
    T profit = std::max(fc, fe);

    if (!(profit > T(0))) return d;

    auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx_best);
    T dy_after_fee = sim.first;

    d.do_trade = true;
    d.i = sel_i;
    d.j = sel_j;
    d.dx = dx_best;
    d.profit = profit;
    d.fee_tokens = sim.second;
    d.notional_coin0 = (sel_i == 0) ? dx_best : dx_best * cex_price;

    return d;
}

} // namespace trading
} // namespace arb
