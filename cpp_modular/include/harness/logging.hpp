// Logging utilities for action recording and detailed output
// Centralizes all save_actions and detailed_log logic
#pragma once

#include <vector>

#include "harness/actions.hpp"
#include "harness/detailed_output.hpp"
#include "pools/twocrypto_fx/helpers.hpp"

namespace arb {
namespace harness {

// ActionLogger: manages action recording when save_actions is enabled
// Provides methods to log each action type, noop when disabled
template <typename T>
class ActionLogger {
public:
    ActionLogger() = default;
    explicit ActionLogger(bool enabled) : enabled_(enabled) {}
    
    bool enabled() const { return enabled_; }
    void set_enabled(bool e) { enabled_ = e; }
    
    // Get recorded actions (moves out)
    std::vector<Action<T>> take_actions() { return std::move(actions_); }
    
    // Get const reference to actions
    const std::vector<Action<T>>& actions() const { return actions_; }
    
    // Log a donation action
    template <typename DonRes, typename DCfg>
    void log_donation(uint64_t ts, const DonRes& don_res, const DCfg& dcfg) {
        if (!enabled_) return;
        DonationAction<T> act;
        act.ts = ts;
        act.ts_due = don_res.ts_due;
        act.amounts = don_res.amounts;
        act.price_scale = don_res.price_scale;
        act.donation_ratio1 = dcfg.ratio1;
        act.apy_per_year = dcfg.apy;
        act.freq_s = dcfg.freq_s;
        actions_.push_back(std::move(act));
    }
    
    // Log a tick action (idle tick with no trade)
    template <typename Pool>
    void log_tick(uint64_t ts, T p_cex, 
                  T ps_before, T oracle_before, T xcp_profit_before, T vp_before,
                  const Pool& pool) {
        if (!enabled_) return;
        TickAction<T> act;
        act.ts = ts;
        act.p_cex = p_cex;
        act.ps_before = ps_before;
        act.ps_after = pool.cached_price_scale;
        act.oracle_before = oracle_before;
        act.oracle_after = pool.cached_price_oracle;
        act.xcp_profit_before = xcp_profit_before;
        act.xcp_profit_after = pool.xcp_profit;
        act.vp_before = vp_before;
        act.vp_after = pool.get_vp_boosted();
        actions_.push_back(std::move(act));
    }
    
    // Log an arb exchange action
    template <typename Pool, typename TW>
    void log_exchange(uint64_t ts, int i, int j, T dx, T dy_after_fee, T fee_tokens,
                      T profit_coin0, T p_cex, T p_pool_before,
                      T oracle_before, T ps_before, uint64_t last_ts_before, T lp_before,
                      T xcp_profit_before, T vp_before,
                      const Pool& pool, const TW& tw) {
        if (!enabled_) return;
        ExchangeAction<T> act;
        act.ts = ts;
        act.i = i;
        act.j = j;
        act.dx = dx;
        act.dy_after_fee = dy_after_fee;
        act.fee_tokens = fee_tokens;
        act.profit_coin0 = profit_coin0;
        act.p_cex = p_cex;
        act.p_pool_before = p_pool_before;
        act.p_pool_after = pool.get_p();
        act.oracle_before = oracle_before;
        act.oracle_after = pool.cached_price_oracle;
        act.ps_before = ps_before;
        act.ps_after = pool.cached_price_scale;
        act.last_ts_before = last_ts_before;
        act.last_ts_after = pool.last_timestamp;
        act.lp_before = lp_before;
        act.lp_after = pool.last_prices;
        act.xcp_profit_before = xcp_profit_before;
        act.xcp_profit_after = pool.xcp_profit;
        act.vp_before = vp_before;
        act.vp_after = pool.get_vp_boosted();
        act.slippage = tw.last_r_inst;
        act.liq_density = tw.last_d_inst;
        act.balance_indicator = pools::twocrypto_fx::balance_indicator(pool);
        actions_.push_back(std::move(act));
    }
    
    // Log a cowswap organic trade action
    void log_cowswap(uint64_t ts, bool is_buy, T dx, T dy_after_fee, T fee_tokens,
                     T hist_dy, T advantage_bps, T threshold_bps,
                     T ps_before, T ps_after) {
        if (!enabled_) return;
        CowswapAction<T> act;
        act.ts = ts;
        act.is_buy = is_buy;
        act.dx = dx;
        act.dy_after_fee = dy_after_fee;
        act.fee_tokens = fee_tokens;
        act.hist_dy = hist_dy;
        act.advantage_bps = advantage_bps;
        act.threshold_bps = threshold_bps;
        act.ps_before = ps_before;
        act.ps_after = ps_after;
        actions_.push_back(std::move(act));
    }

private:
    bool enabled_{false};
    std::vector<Action<T>> actions_;
};

// DetailedLogger: logs pool state at every N-th event when enabled
template <typename T>
class DetailedLogger {
public:
    DetailedLogger() = default;
    explicit DetailedLogger(bool enabled, size_t interval = 1) 
        : enabled_(enabled), interval_(interval > 0 ? interval : 1) {}
    
    bool enabled() const { return enabled_; }
    
    // Get recorded entries (moves out)
    std::vector<DetailedEntry<T>> take_entries() { return std::move(entries_); }
    
    // Log current pool state for this event (respects interval)
    template <typename Pool>
    void log_event(const Pool& pool, uint64_t ts, const Candle& candle, uint64_t n_trades, uint64_t n_rebalances) {
        if (!enabled_) return;
        
        // Only log every interval_ events
        if (event_count_++ % interval_ != 0) return;
        
        DetailedEntry<T> entry;
        entry.t = ts;
        entry.token0 = pool.balances[0];
        entry.token1 = pool.balances[1];
        entry.price_oracle = pool.cached_price_oracle;
        entry.price_scale = pool.cached_price_scale;
        entry.profit = pool.get_virtual_price() - T(1);
        entry.xcp = pool.xcp_profit;
        entry.open = static_cast<T>(candle.open);
        entry.high = static_cast<T>(candle.high);
        entry.low = static_cast<T>(candle.low);
        entry.close = static_cast<T>(candle.close);
        const auto xp_now = pools::twocrypto_fx::pool_xp_current(pool);
        entry.fee = pools::twocrypto_fx::dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);
        entry.n_trades = n_trades;
        entry.n_rebalances = n_rebalances;
        entries_.push_back(entry);
    }

private:
    bool enabled_{false};
    size_t interval_{1};
    size_t event_count_{0};
    std::vector<DetailedEntry<T>> entries_;
};

} // namespace harness
} // namespace arb
