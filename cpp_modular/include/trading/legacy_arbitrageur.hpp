// Legacy arbitrageur - matches simusmod step_for_price_2 sizing logic
// For compatibility comparison with legacy simulator
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <iomanip>
#include <string>

#include "pools/twocrypto_fx/helpers.hpp"
#include "trading/costs.hpp"
#include "trading/decision.hpp"

namespace arb {
namespace trading {

namespace fx = arb::pools::twocrypto_fx;

// Debug flag - set TRACE_LEGACY_ARB=1 to enable verbose output
inline bool trace_legacy_arb_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("TRACE_LEGACY_ARB");
        return env && std::string(env) == "1";
    }();
    return enabled;
}

// Legacy arbitrageur that matches simusmod step_for_price_2 behavior exactly
// Key differences from modular:
// - Uses iterative doubling/halving to find optimal size (not root-finding)
// - Maximizes profit directly rather than finding price equilibrium
// - Uses ext_fee as CEX fee (not arb_fee_bps)
// - Volume cap based on candle volume
template <typename T, typename PoolT>
Decision<T> decide_trade_legacy(
    const PoolT& pool,
    T cex_price,           // Current CEX price (high or low from candle)
    const Costs<T>& costs,
    T ext_vol,             // External volume from candle (in coin0 terms)
    [[maybe_unused]] T min_swap_frac,  // Unused in legacy (kept for interface compatibility)
    T max_swap_frac        // Used for balance limit check
) {
    static_assert(std::is_floating_point_v<T>, "decide_trade_legacy is floating-only");

    Decision<T> d{};

    if (!(cex_price > T(0))) return d;

    const T ext_fee = costs.arb_fee_bps / T(10000);  // Legacy uses ext_fee
    const T gas_fee = costs.gas_coin0;

    // Get current pool price (marginal price for tiny trade)
    // Legacy: price_2(a, b) uses dx = D * 1e-8 as step
    const T D_val = pool.D;
    const T dx_marginal = D_val * T(1e-8);
    
    // Compute current pool price via simulation of tiny trade (WITHOUT fees, like legacy price_2)
    // Legacy price_2 computes: dx_raw / (curve.x[j] - curve_res) with no fee adjustment
    auto get_pool_price = [&]() -> T {
        // Simulate without fees - get raw dy from invariant
        const T ps = pool.cached_price_scale;
        auto balances_local = pool.balances;
        balances_local[0] += dx_marginal;
        auto xp = fx::pool_xp_from(pool, balances_local, ps);
        auto y_out = fx::MathOps<T>::get_y(pool.A, pool.gamma, xp, pool.D, 1);
        T dy_xp = xp[1] - y_out.value;
        T dy_tokens = fx::xp_to_tokens_j(pool, 1, dy_xp, ps);
        if (dy_tokens <= T(0)) return T(0);
        return dx_marginal / dy_tokens;  // price in coin0/coin1 WITHOUT fees
    };
    
    T p_pool = get_pool_price();
    if (!(p_pool > T(0))) return d;

    // Determine direction:
    // max_price path: CEX high > pool price -> buy coin1 on pool (sell coin0), sell coin1 on CEX
    // min_price path: CEX low < pool price -> buy coin1 on CEX, sell coin1 on pool (buy coin0)
    
    T max_price = cex_price * (T(1) - ext_fee);  // Effective price when selling to CEX
    T min_price = cex_price * (T(1) + ext_fee);  // Effective price when buying from CEX

    // Legacy processes max_price first (if cex high > pool), then min_price (if cex low < pool)
    // We'll check which direction has an opportunity
    
    bool try_buy_coin1 = (max_price > p_pool);   // Buy coin1 on pool, sell on CEX
    bool try_sell_coin1 = (min_price < p_pool);  // Buy coin1 on CEX, sell on pool

    // Volume cap: legacy computes ext_vol = d.volume * price_oracle[b] (convert ETH volume to USD)
    // Then uses ext_vol/2 as cap in step_for_price_2.
    // Our events already split volume in half (so ext_vol here is candle_vol/2 in ETH).
    // Legacy also divides by 2 in the trade check, so effective cap = candle_vol/4.
    // We need to: convert to USD, then divide by 2 to match legacy.
    const T price_oracle_b = (pool.last_prices > T(0)) ? pool.last_prices : cex_price;
    const T ext_vol_usd = ext_vol * price_oracle_b;  // Convert from ETH to USD
    const T vol_cap = ext_vol_usd / T(2);  // Divide by 2 to match legacy step_for_price_2

    if (trace_legacy_arb_enabled()) {
        std::cerr << "[LEGACY_ARB] cex_price=" << cex_price << " ext_fee=" << ext_fee 
                  << " p_pool=" << p_pool << " max_price=" << max_price << " min_price=" << min_price
                  << " try_buy=" << try_buy_coin1 << " try_sell=" << try_sell_coin1 
                  << " ext_vol=" << ext_vol << " vol_cap=" << vol_cap
                  << " D=" << pool.D << " ps=" << pool.cached_price_scale 
                  << " b0=" << pool.balances[0] << " b1=" << pool.balances[1] << "\n";
    }

    if (!try_buy_coin1 && !try_sell_coin1) return d;

    // Helper to get step0 for a given input coin
    auto get_step0 = [&](int i) -> T {
        if (i == 0) return dx_marginal;  // coin0: step in USDC
        return dx_marginal / pool.cached_price_scale;  // coin1: step in ETH
    };

    // Helper to compute profit for a given trade
    auto compute_profit_and_price = [&](int i, int j, T _dx, T target_price, bool is_buy) 
        -> std::pair<T, T> {
        // Simulate the exchange
        auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(i), static_cast<size_t>(j), _dx);
        T dy_after_fee = sim.first;
        
        if (dy_after_fee <= T(0)) return {T(-1e30), T(0)};
        
        // Compute effective price
        T price = (i == 0) ? (_dx / dy_after_fee) : (dy_after_fee / _dx);
        
        // Compute profit (legacy formula)
        T profit;
        if (is_buy) {
            // Buying coin1: profit = (dx/price - dx/max_price) * max_price
            // = dx * (max_price/price - 1)
            profit = (_dx / price - _dx / target_price) * target_price;
        } else {
            // Selling coin1: profit = (price - min_price) * dx
            profit = (price - target_price) * _dx;
        }
        
        if (trace_legacy_arb_enabled() && _dx < dx_marginal * T(10)) {
            std::cerr << "[LEGACY_ARB] compute_profit: i=" << i << " j=" << j 
                      << " dx=" << _dx << " dy=" << dy_after_fee << " price=" << price 
                      << " target=" << target_price << " profit=" << profit << "\n";
        }
        
        return {profit, price};
    };

    // Try buy_coin1 direction (i=0, j=1)
    auto try_direction = [&](int i, int j, T target_price, bool is_buy) -> Decision<T> {
        Decision<T> result{};
        
        const T step0_i = get_step0(i);  // Step size depends on input coin
        T _dx = T(0);
        T step = step0_i;
        T previous_profit = T(0);
        
        const T balance_i = pool.balances[static_cast<size_t>(i)];
        
        if (trace_legacy_arb_enabled()) {
            std::cerr << "[LEGACY_ARB] try_direction i=" << i << " j=" << j 
                      << " target_price=" << target_price << " is_buy=" << is_buy 
                      << " balance_i=" << balance_i << " step0=" << step0_i << "\n";
        }
        
        // Phase 1: Exponential increase (step doubles each iteration)
        while (true) {
            T _dx_prev = _dx;
            _dx += step;
            
            // Check balance limit
            if (_dx > balance_i * max_swap_frac) {
                _dx = _dx_prev;
                break;
            }
            
            auto [new_profit, price] = compute_profit_and_price(i, j, _dx, target_price, is_buy);
            
            // Compute volume in coin0 terms using spot price (not price_scale)
            auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(i), static_cast<size_t>(j), _dx);
            T dy = sim.first;
            const T price_ref = (pool.last_prices > T(0)) ? pool.last_prices : cex_price;
            const T v = (j == 1) ? (dy * price_ref) : dy;
            
            if (trace_legacy_arb_enabled() && _dx < step0_i * T(100)) {
                std::cerr << "[LEGACY_ARB] P1: _dx=" << _dx << " profit=" << new_profit 
                          << " v=" << v << " vol_cap=" << vol_cap << " prev_profit=" << previous_profit << "\n";
            }
            
            if (new_profit > previous_profit && v <= vol_cap) {
                previous_profit = new_profit;
            } else {
                _dx = _dx_prev;
                break;
            }
            
            step += step;  // Double the step
        }
        
        // Phase 2: Binary search refinement (step halves, try +/- directions)
        while (true) {
            T _dx_prev = _dx;
            if (step < T(0)) step = -step;
            step /= T(2);
            
            if (step < step0_i) break;
            
            for (int ctr = 0; ctr < 2; ctr++) {
                step = -step;
                T _dx_try = _dx_prev + step;
                
                if (_dx_try <= T(0) || _dx_try > balance_i * max_swap_frac) continue;
                
                auto [new_profit, price] = compute_profit_and_price(i, j, _dx_try, target_price, is_buy);
                
                auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(i), static_cast<size_t>(j), _dx_try);
                T dy = sim.first;
                const T price_ref = (pool.last_prices > T(0)) ? pool.last_prices : cex_price;
                const T v = (j == 1) ? (dy * price_ref) : dy;
                
                if (new_profit > previous_profit && v <= vol_cap) {
                    previous_profit = new_profit;
                    _dx = _dx_try;
                    break;
                }
            }
        }
        
        if (_dx <= T(0)) return result;
        
        // Final gas check (legacy does this at the end)
        T gas_in_coin_i = (i == 0) ? gas_fee : (gas_fee / pool.cached_price_scale);
        
        // Recalculate profit with gas
        T profit_with_gas;
        if (is_buy) {
            // price_with_gas = (_dx + gas) / _dy
            auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(i), static_cast<size_t>(j), _dx);
            T dy = sim.first;
            T price_with_gas = (_dx + gas_in_coin_i) / dy;
            profit_with_gas = (_dx / price_with_gas - _dx / target_price) * target_price;
        } else {
            auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(i), static_cast<size_t>(j), _dx);
            T dy = sim.first;
            T price_with_gas = dy / (_dx + gas_in_coin_i);
            profit_with_gas = (price_with_gas - target_price) * _dx;
        }
        
        if (profit_with_gas <= T(0)) {
            if (trace_legacy_arb_enabled()) {
                std::cerr << "[LEGACY_ARB] Gas check failed: profit_with_gas=" << profit_with_gas << "\n";
            }
            return result;
        }
        
        // Trade is profitable
        auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(i), static_cast<size_t>(j), _dx);
        
        result.do_trade = true;
        result.i = i;
        result.j = j;
        result.dx = _dx;
        result.profit = profit_with_gas;
        result.fee_tokens = sim.second;
        const T price_ref = (pool.last_prices > T(0)) ? pool.last_prices : cex_price;
        result.notional_coin0 = (i == 0) ? _dx : _dx * price_ref;
        
        if (trace_legacy_arb_enabled()) {
            std::cerr << "[LEGACY_ARB] TRADE: i=" << i << " j=" << j 
                      << " dx=" << _dx << " profit=" << profit_with_gas << "\n";
        }
        
        return result;
    };

    // Try buy_coin1 first (matches legacy order)
    if (try_buy_coin1) {
        auto result = try_direction(0, 1, max_price, true);
        if (result.do_trade) return result;
    }
    
    // Then try sell_coin1
    if (try_sell_coin1) {
        auto result = try_direction(1, 0, min_price, false);
        if (result.do_trade) return result;
    }

    return d;
}

} // namespace trading
} // namespace arb
