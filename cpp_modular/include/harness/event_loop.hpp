// Event loop for processing price events and executing arb trades
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "core/common.hpp"
#include "events/types.hpp"
#include "harness/metrics.hpp"
#include "harness/actions.hpp"
#include "harness/detailed_output.hpp"
#include "harness/logging.hpp"
#include "harness/donation.hpp"
#include "harness/idle_tick.hpp"
#include "harness/user_swap.hpp"
#include "trading/costs.hpp"
#include "trading/decision.hpp"
#include "trading/arbitrageur.hpp"
#include "trading/cowswap_trader.hpp"
#include "pools/twocrypto_fx/helpers.hpp"

namespace arb {
namespace harness {

// Process events and execute trades with full metrics sampling.
// Returns EventLoopResult with all metrics.
template <typename T, typename Pool>
EventLoopResult<T> run_event_loop(
    Pool& pool,
    const std::vector<Event>& events,
    const trading::Costs<T>& costs,
    DonationCfg<T>& dcfg,
    IdleTickCfg<T>& icfg,
    UserSwapCfg<T>& ucfg,
    T min_swap_frac = T(1e-6),
    T max_swap_frac = T(1.0),
    size_t max_events = 0,  // 0 = all events
    bool enable_slippage_probes = true,
    bool save_actions = false,
    bool detailed_log = false,
    size_t detailed_interval = 1,  // log every N-th event (1 = all)
    trading::CowswapTrader<T>* cowswap = nullptr  // Optional cowswap trader
) {
    EventLoopResult<T> result{};
    Metrics<T>& m = result.metrics;
    TimeWeightedMetrics<T>& tw = result.tw_metrics;
    SlippageProbes<T>& sp = result.slippage_probes;
    
    const size_t n_events = max_events > 0 ? std::min(max_events, events.size()) : events.size();
    if (n_events == 0) return result;
    
    // Record timestamps
    result.t_start = events.front().ts;
    result.t_end = events[n_events - 1].ts;
    
    // Record initial state
    result.tvl_start = pool.balances[0] + pool.balances[1] * pool.cached_price_scale;
    result.donation_apy = dcfg.apy;
    
    // Probe sizes for slippage: 1%, 5%, 10% of initial TVL (coin0 terms)
    std::array<T, SlippageProbes<T>::N_SIZES> probe_sizes_coin0{};
    if (enable_slippage_probes) {
        for (size_t k = 0; k < SlippageProbes<T>::N_SIZES; ++k) {
            probe_sizes_coin0[k] = result.tvl_start * static_cast<T>(SlippageProbes<T>::SIZE_FRACS[k]);
        }
    }
    
    // Initialize loggers
    ActionLogger<T> action_logger(save_actions);
    DetailedLogger<T> detailed_logger(detailed_log, detailed_interval);
    
    // Helper to sample slippage probes
    auto sample_slippage_probes = [&](uint64_t ts, T p_cex) {
        if (!enable_slippage_probes || !(p_cex > T(0))) return;
        for (size_t k = 0; k < SlippageProbes<T>::N_SIZES; ++k) {
            sp.accumulate_previous(k, ts);
            
            const T S = probe_sizes_coin0[k];
            // 0 -> 1
            {
                auto pr = pools::twocrypto_fx::simulate_exchange_once(pool, 0, 1, S);
                const T dy1 = pr.first;  // coin1 after fee
                const T ideal1 = S / p_cex;
                T s01 = T(0);
                if (ideal1 > T(0)) {
                    s01 = T(1) - (dy1 / ideal1);
                }
                // 1 -> 0
                const T dx1 = S / p_cex;
                auto pr10 = pools::twocrypto_fx::simulate_exchange_once(pool, 1, 0, dx1);
                const T dy0 = pr10.first;  // coin0 after fee
                T s10 = T(0);
                if (S > T(0)) {
                    s10 = T(1) - (dy0 / S);
                }
                sp.sample(k, ts, s01, s10);
            }
        }
    };
    
    auto sample_pre_trade = [&](uint64_t ts, T cex_price) {
        tw.sample_price_error(ts, pool.cached_price_scale, cex_price);

        const T x0p = pool.balances[0];
        const T x1p = pool.balances[1] * cex_price;
        tw.sample_imbalance(ts, x0p, x1p);

        const auto xp_now = pools::twocrypto_fx::pool_xp_current(pool);
        const T cur_fee = pools::twocrypto_fx::dyn_fee(
            xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma
        );
        tw.sample_fee(ts, cur_fee);
    };

    auto apply_donation = [&](uint64_t ts) {
        auto don_res = make_donation_ex(pool, dcfg, ts, m);
        if (don_res.success) {
            action_logger.log_donation(ts, don_res, dcfg);
        }
    };

    auto apply_user_swap = [&](uint64_t ts, T cex_price) {
        try_user_swap(pool, ucfg, ts, cex_price);
    };

    auto execute_arb = [&](const Event& ev, T cex_price) -> bool {
        T volume_cap = std::numeric_limits<T>::infinity();
        if (costs.use_volume_cap) {
            volume_cap = static_cast<T>(ev.volume) * costs.volume_cap_mult;
            if (!costs.volume_cap_is_coin1) {
                volume_cap *= cex_price;
            }
        }

        auto dec = trading::decide_trade(
            pool, cex_price, costs,
            volume_cap,
            min_swap_frac, max_swap_frac
        );
        if (!dec.do_trade) {
            return false;
        }

        try {
            const T ps_before = pool.cached_price_scale;
            const T oracle_before = pool.cached_price_oracle;
            const T xcp_profit_before = pool.xcp_profit;
            const T vp_before = pool.get_vp_boosted();
            const T p_pool_before = pool.get_p();
            const uint64_t last_ts_before = pool.last_timestamp;
            const T lp_before = pool.last_prices;

            auto res = pool.exchange(
                static_cast<T>(dec.i),
                static_cast<T>(dec.j),
                dec.dx,
                T(0)
            );

            const T dy_after_fee = res[0];
            const T fee_tokens = res[1];
            const T ps_after = pool.cached_price_scale;

            m.trades += 1;
            m.notional += dec.notional_coin0;
            m.lp_fee_coin0 += (dec.j == 1 ? fee_tokens * cex_price : fee_tokens);
            m.arb_pnl_coin0 += dec.profit;

            const T gross_dy_tokens = dy_after_fee + fee_tokens;
            if (gross_dy_tokens > T(0) && dec.notional_coin0 > T(0)) {
                const T fee_frac = fee_tokens / gross_dy_tokens;
                m.fee_wsum += fee_frac * dec.notional_coin0;
                m.fee_w += dec.notional_coin0;
            }

            if (differs_rel(ps_after, ps_before)) {
                m.n_rebalances += 1;
            }

            sample_slippage_probes(ev.ts, cex_price);

            action_logger.log_exchange(ev.ts, dec.i, dec.j, dec.dx, dy_after_fee, fee_tokens,
                                       dec.profit, cex_price, p_pool_before,
                                       oracle_before, ps_before, last_ts_before, lp_before,
                                       xcp_profit_before, vp_before, pool, tw);
            return true;
        } catch (...) {
            return false;
        }
    };

    auto apply_cowswap = [&]() -> bool {
        if (!(cowswap && cowswap->has_pending())) {
            return false;
        }
        trading::CowswapMetrics<T> cs_metrics{};
        std::vector<trading::CowswapExecDetail<T>> cs_exec_details;
        std::vector<trading::CowswapExecDetail<T>>* cs_details_ptr =
            action_logger.enabled() ? &cs_exec_details : nullptr;

        cowswap->apply_due_trades(pool, cs_metrics, cs_details_ptr);

        m.cowswap_trades += cs_metrics.trades_executed;
        m.cowswap_skipped += cs_metrics.trades_skipped;
        m.cowswap_notional_coin0 += cs_metrics.notional_coin0;
        m.cowswap_lp_fee_coin0 += cs_metrics.lp_fee_coin0;

        if (action_logger.enabled() && !cs_exec_details.empty()) {
            for (const auto& detail : cs_exec_details) {
                action_logger.log_cowswap(detail.ts, detail.is_buy, detail.dx,
                                          detail.dy_after_fee, detail.fee_tokens,
                                          detail.hist_dy, detail.advantage_bps, detail.threshold_bps,
                                          detail.ps_before, detail.ps_after);
            }
        }

        return cs_metrics.trades_executed > 0;
    };

    auto apply_idle_tick = [&](uint64_t ts, T cex_price) -> bool {
        const T ps_before = pool.cached_price_scale;
        const T oracle_before = pool.cached_price_oracle;
        const T xcp_profit_before = pool.xcp_profit;
        const T vp_before = pool.get_vp_boosted();

        const bool did_tick = try_idle_tick(pool, icfg, ts, m);
        if (!did_tick) {
            return false;
        }

        sample_slippage_probes(ts, cex_price);
        action_logger.log_tick(ts, cex_price, ps_before, oracle_before,
                               xcp_profit_before, vp_before, pool);
        return true;
    };

    for (size_t ev_idx = 0; ev_idx < n_events; ++ev_idx) {
        const auto& ev = events[ev_idx];

        pool.set_block_timestamp(ev.ts);
        const T cex_price = static_cast<T>(ev.p_cex);

        sample_pre_trade(ev.ts, cex_price);
        apply_donation(ev.ts);

        if (!(cex_price > T(0))) {
            continue;
        }

        apply_user_swap(ev.ts, cex_price);

        bool did_any_trade = execute_arb(ev, cex_price);
        if (apply_cowswap()) {
            did_any_trade = true;
        }
        if (!did_any_trade) {
            did_any_trade = apply_idle_tick(ev.ts, cex_price);
        }

        detailed_logger.log_event(pool, ev.ts, ev.candle, cex_price, m.trades, m.n_rebalances);
    }
    
    // Move logged data into result
    result.actions = action_logger.take_actions();
    result.detailed_entries = detailed_logger.take_entries();
    
    return result;
}

// Backward-compatible overload returning just Metrics<T>
template <typename T, typename Pool>
Metrics<T> run_event_loop_simple(
    Pool& pool,
    const std::vector<Event>& events,
    const trading::Costs<T>& costs,
    DonationCfg<T>& dcfg,
    IdleTickCfg<T>& icfg,
    UserSwapCfg<T>& ucfg,
    T min_swap_frac = T(1e-6),
    T max_swap_frac = T(1.0),
    size_t max_events = 0,
    bool enable_slippage_probes = true
) {
    auto result = run_event_loop(pool, events, costs, dcfg, icfg, ucfg,
                                  min_swap_frac, max_swap_frac, max_events,
                                  enable_slippage_probes);
    return result.metrics;
}

} // namespace harness
} // namespace arb
