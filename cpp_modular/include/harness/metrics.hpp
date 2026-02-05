// Metrics tracking for arbitrage harness
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "harness/actions.hpp"
#include "harness/detailed_output.hpp"

namespace arb {
namespace harness {

// Core trading metrics
template <typename T>
struct Metrics {
    // Trade execution
    size_t trades{0};
    T notional{0};              // Total notional in coin0 units
    T lp_fee_coin0{0};          // Total LP fees in coin0 units
    T arb_pnl_coin0{0};         // Arbitrageur profit in coin0 units
    size_t n_rebalances{0};     // Count of price_scale changes
    
    // Donations
    size_t donations{0};
    T donation_coin0_total{0};
    std::array<T, 2> donation_amounts_total{T(0), T(0)};
    
    // Cowswap organic trades
    size_t cowswap_trades{0};
    size_t cowswap_skipped{0};
    T cowswap_notional_coin0{0};
    T cowswap_lp_fee_coin0{0};
    
    // Size-weighted average pool fee tracking
    T fee_wsum{0};   // sum(fee_fraction * notional_coin0)
    T fee_w{0};      // sum(notional_coin0)
    
    // Computed metrics
    T avg_pool_fee() const {
        return fee_w > T(0) ? fee_wsum / fee_w : T(-1);
    }
    
    // Accumulate another metrics instance
    void accumulate(const Metrics& other) {
        trades += other.trades;
        notional += other.notional;
        lp_fee_coin0 += other.lp_fee_coin0;
        arb_pnl_coin0 += other.arb_pnl_coin0;
        n_rebalances += other.n_rebalances;
        donations += other.donations;
        donation_coin0_total += other.donation_coin0_total;
        donation_amounts_total[0] += other.donation_amounts_total[0];
        donation_amounts_total[1] += other.donation_amounts_total[1];
        cowswap_trades += other.cowswap_trades;
        cowswap_skipped += other.cowswap_skipped;
        cowswap_notional_coin0 += other.cowswap_notional_coin0;
        cowswap_lp_fee_coin0 += other.cowswap_lp_fee_coin0;
        fee_wsum += other.fee_wsum;
        fee_w += other.fee_w;
    }
};

struct TimeWeightedSummary {
    double avg_rel_price_diff{-1.0};
    double max_rel_price_diff{-1.0};
    double avg_imbalance{-1.0};
    double tw_avg_pool_fee{-1.0};
};

// Time-weighted metrics for tracking pool state over time
template <typename T>
struct TimeWeightedMetrics {
    // Price follow metrics (relative): time-weighted |ps/p_cex - 1|
    long double sum_abs_rel_dt{0.0L};   // sum |rel_err| * dt
    long double sum_dt{0.0L};            // sum dt
    long double max_rel_abs{0.0L};       // max |rel_err| across events
    T last_rel_abs{0};                   // |ps/p_cex - 1| at previous event
    uint64_t last_ts_err{0};
    bool have_err{false};

    // Time-weighted imbalance: 4*x0'*x1'/(x0'+x1')^2, with x1' = balance1 * p_cex
    long double sum_imbalance_dt{0.0L};
    long double imbalance_dt{0.0L};
    T last_imbalance{0};
    uint64_t last_ts_imbalance{0};
    bool have_imbalance{false};
    
    // Time-weighted pool fee (fraction) across time
    long double tw_fee_sum_dt{0.0L};
    long double tw_fee_dt{0.0L};
    T last_fee_frac{0};
    uint64_t last_ts_fee{0};
    bool have_fee{false};
    
    
    TimeWeightedSummary summarize() const {
        TimeWeightedSummary summary{};
        summary.avg_rel_price_diff = sum_dt > 0.0L
            ? static_cast<double>(sum_abs_rel_dt / sum_dt)
            : -1.0;
        summary.max_rel_price_diff = static_cast<double>(max_rel_abs);
        summary.avg_imbalance = imbalance_dt > 0.0L
            ? static_cast<double>(sum_imbalance_dt / imbalance_dt)
            : -1.0;
        summary.tw_avg_pool_fee = tw_fee_dt > 0.0L
            ? static_cast<double>(tw_fee_sum_dt / tw_fee_dt)
            : -1.0;
        return summary;
    }

    // Update methods
    void sample_price_error(uint64_t ts, T price_scale, T p_cex) {
        T cur_rel_abs = T(0);
        if (p_cex > T(0)) {
            cur_rel_abs = std::abs(price_scale / p_cex - T(1));
        }
        
        if (have_err && ts > last_ts_err) {
            const long double dt = static_cast<long double>(ts - last_ts_err);
            sum_abs_rel_dt += static_cast<long double>(last_rel_abs) * dt;
            sum_dt += dt;
        }
        
        if (static_cast<long double>(cur_rel_abs) > max_rel_abs) {
            max_rel_abs = static_cast<long double>(cur_rel_abs);
        }
        
        last_rel_abs = cur_rel_abs;
        last_ts_err = ts;
        have_err = true;
    }

    void sample_imbalance(uint64_t ts, T x0p, T x1p) {
        T cur = T(0);
        const T denom = x0p + x1p;
        if (denom > T(0)) {
            cur = (T(4) * x0p * x1p) / (denom * denom);
        }

        if (have_imbalance && ts > last_ts_imbalance) {
            const long double dt = static_cast<long double>(ts - last_ts_imbalance);
            sum_imbalance_dt += static_cast<long double>(last_imbalance) * dt;
            imbalance_dt += dt;
        }

        last_imbalance = cur;
        last_ts_imbalance = ts;
        have_imbalance = true;
    }
    
    void sample_fee(uint64_t ts, T fee_frac) {
        if (have_fee && ts > last_ts_fee) {
            const long double dt = static_cast<long double>(ts - last_ts_fee);
            tw_fee_sum_dt += static_cast<long double>(last_fee_frac) * dt;
            tw_fee_dt += dt;
        }
        last_fee_frac = fee_frac;
        last_ts_fee = ts;
        have_fee = true;
    }
    
};

// Real slippage probes at fixed sizes
template <typename T>
struct SlippageProbes {
    static constexpr size_t N_SIZES = 3;
    static constexpr double SIZE_FRACS[N_SIZES] = {0.01, 0.05, 0.10};  // 1%, 5%, 10%
    
    std::array<long double, N_SIZES> tw_real_s01_sum_dt{};  // 0->1 direction
    std::array<long double, N_SIZES> tw_real_s10_sum_dt{};  // 1->0 direction
    std::array<long double, N_SIZES> tw_real_dt{};
    std::array<T, N_SIZES> last_real_s01{};
    std::array<T, N_SIZES> last_real_s10{};
    std::array<uint64_t, N_SIZES> last_ts_real{};
    std::array<bool, N_SIZES> have_real{};
    
    SlippageProbes() {
        for (size_t k = 0; k < N_SIZES; ++k) {
            tw_real_s01_sum_dt[k] = 0.0L;
            tw_real_s10_sum_dt[k] = 0.0L;
            tw_real_dt[k] = 0.0L;
            last_real_s01[k] = T(0);
            last_real_s10[k] = T(0);
            last_ts_real[k] = 0;
            have_real[k] = false;
        }
    }
    
    void accumulate_previous(size_t k, uint64_t ts) {
        if (k >= N_SIZES) return;
        if (have_real[k] && ts > last_ts_real[k]) {
            const long double dt = static_cast<long double>(ts - last_ts_real[k]);
            tw_real_s01_sum_dt[k] += static_cast<long double>(last_real_s01[k]) * dt;
            tw_real_s10_sum_dt[k] += static_cast<long double>(last_real_s10[k]) * dt;
            tw_real_dt[k] += dt;
        }
    }
    
    void sample(size_t k, uint64_t ts, T s01, T s10) {
        if (k >= N_SIZES) return;
        last_real_s01[k] = s01;
        last_real_s10[k] = s10;
        last_ts_real[k] = ts;
        have_real[k] = true;
    }
    
    double tw_slippage(size_t k) const {
        if (k >= N_SIZES || tw_real_dt[k] <= 0.0L) return -1.0;
        return static_cast<double>((tw_real_s01_sum_dt[k] + tw_real_s10_sum_dt[k]) / (2.0L * tw_real_dt[k]));
    }
};

// Complete result from event loop including all metrics
template <typename T>
struct EventLoopResult {
    // Core trading metrics
    Metrics<T> metrics{};
    
    // Time-weighted metrics
    TimeWeightedMetrics<T> tw_metrics{};
    
    // Slippage probes
    SlippageProbes<T> slippage_probes{};
    
    // Actions (only populated if save_actions=true)
    std::vector<Action<T>> actions{};
    
    // Detailed per-candle log (only populated if detailed logging enabled)
    std::vector<DetailedEntry<T>> detailed_entries{};
    
    // Start/end timestamps for duration calculation
    uint64_t t_start{0};
    uint64_t t_end{0};
    
    // Initial state for APY calculations
    T tvl_start{0};
    
    // Donation APY (for net calculations)
    T donation_apy{0};
    
    double duration_s() const {
        return (t_end > t_start) ? static_cast<double>(t_end - t_start) : 0.0;
    }
};

} // namespace harness
} // namespace arb
