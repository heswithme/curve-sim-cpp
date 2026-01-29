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
    
    // Time-weighted pool fee (fraction) across time
    long double tw_fee_sum_dt{0.0L};
    long double tw_fee_dt{0.0L};
    T last_fee_frac{0};
    uint64_t last_ts_fee{0};
    bool have_fee{false};
    
    // Time-weighted latent slippage/liquidity density
    long double tw_d_sum_dt{0.0L};  // sum d * dt
    long double tw_r_sum_dt{0.0L};  // sum r * dt
    long double tw_dt{0.0L};         // total dt
    T last_d_inst{0};
    T last_r_inst{0};
    uint64_t last_ts_lat{0};
    bool have_lat{false};
    
    // Fraction of time price_scale deviates more than 10% from p_cex
    static constexpr T FAR_THRESH = T(0.10);
    long double time_far_s{0.0L};
    uint64_t last_ts_band{0};
    bool have_band{false};
    bool last_far{false};
    
    // Multi-threshold band tracking: time within X% of CEX
    // Thresholds: 3%, 5%, 10%, 20%, 30%, and dynamic 1/A
    static constexpr size_t N_THRESH = 6;
    static constexpr double THRESH_VALUES[N_THRESH] = {0.03, 0.05, 0.10, 0.20, 0.30, 0.0};  // last is dynamic 1/A
    std::array<long double, N_THRESH> time_within_s{};  // accumulated time within each threshold
    std::array<bool, N_THRESH> last_within{};
    uint64_t last_ts_thresh{0};
    bool have_thresh{false};
    
    // EMA-smoothed correlation between cex_price and price_scale
    // tau_s = smoothing window in seconds (default 1 hour)
    static constexpr double CORR_TAU_S = 86400.0;
    long double ema_cex{0.0L};
    long double ema_ps{0.0L};
    long double sum_x_dt{0.0L};    // Σ ema_cex * dt
    long double sum_y_dt{0.0L};    // Σ ema_ps * dt
    long double sum_xx_dt{0.0L};   // Σ ema_cex² * dt
    long double sum_yy_dt{0.0L};   // Σ ema_ps² * dt
    long double sum_xy_dt{0.0L};   // Σ ema_cex * ema_ps * dt
    long double sum_corr_dt{0.0L}; // Σ dt for correlation
    uint64_t last_ts_corr{0};
    bool have_corr{false};
    
    // Computed metrics
    double avg_rel_price_diff() const {
        return sum_dt > 0.0L ? static_cast<double>(sum_abs_rel_dt / sum_dt) : -1.0;
    }
    
    double max_rel_price_diff() const {
        return static_cast<double>(max_rel_abs);
    }
    
    double tw_avg_pool_fee() const {
        return tw_fee_dt > 0.0L ? static_cast<double>(tw_fee_sum_dt / tw_fee_dt) : -1.0;
    }
    
    // Normalized vs xy=k baseline
    // For xy=k: r = 2, d = 1/2 => tw_slippage ≈ 1, tw_liq_density ≈ 1
    double tw_slippage() const {
        if (tw_dt <= 0.0L) return -1.0;
        return static_cast<double>(tw_r_sum_dt / tw_dt) / 2.0;
    }
    
    double tw_liq_density() const {
        if (tw_dt <= 0.0L) return -1.0;
        return static_cast<double>(tw_d_sum_dt / tw_dt) * 2.0;
    }
    
    double cex_follow_time_frac(double duration_s) const {
        if (duration_s <= 0.0) return -1.0;
        return 1.0 - static_cast<double>(time_far_s) / duration_s;
    }
    
    // Get fraction of time within threshold k (0=3%, 1=5%, 2=10%, 3=20%, 4=30%, 5=1/A)
    double pct_within(size_t k, double duration_s) const {
        if (k >= N_THRESH || duration_s <= 0.0) return -1.0;
        return static_cast<double>(time_within_s[k]) / duration_s;
    }
    
    // Get EMA-smoothed correlation between cex_price and price_scale
    double price_correlation() const {
        if (sum_corr_dt <= 0.0L) return -2.0;  // not enough data
        
        const long double E_x = sum_x_dt / sum_corr_dt;
        const long double E_y = sum_y_dt / sum_corr_dt;
        const long double E_xx = sum_xx_dt / sum_corr_dt;
        const long double E_yy = sum_yy_dt / sum_corr_dt;
        const long double E_xy = sum_xy_dt / sum_corr_dt;
        
        const long double var_x = E_xx - E_x * E_x;
        const long double var_y = E_yy - E_y * E_y;
        const long double cov_xy = E_xy - E_x * E_y;
        
        if (var_x <= 0.0L || var_y <= 0.0L) return -2.0;  // no variance
        
        return static_cast<double>(cov_xy / std::sqrt(var_x * var_y));
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
    
    void sample_latent(uint64_t ts, T d_inst, T r_inst) {
        if (have_lat && ts > last_ts_lat) {
            const long double dt = static_cast<long double>(ts - last_ts_lat);
            tw_d_sum_dt += static_cast<long double>(last_d_inst) * dt;
            tw_r_sum_dt += static_cast<long double>(last_r_inst) * dt;
            tw_dt += dt;
        }
        last_d_inst = d_inst;
        last_r_inst = r_inst;
        last_ts_lat = ts;
        have_lat = true;
    }
    
    void sample_band(uint64_t ts, T price_scale, T p_cex) {
        if (have_band && ts > last_ts_band && last_far) {
            time_far_s += static_cast<long double>(ts - last_ts_band);
        }
        
        if (p_cex > T(0)) {
            const T rel = std::abs(price_scale / p_cex - T(1));
            last_far = (rel > FAR_THRESH);
        } else {
            last_far = false;
        }
        last_ts_band = ts;
        have_band = true;
    }
    
    // Sample multi-threshold bands: track time within 3%, 5%, 10%, 20%, 30%, and 1/A of CEX
    void sample_thresholds(uint64_t ts, T price_scale, T p_cex, T A) {
        // Accumulate time for previous state
        if (have_thresh && ts > last_ts_thresh) {
            const long double dt = static_cast<long double>(ts - last_ts_thresh);
            for (size_t k = 0; k < N_THRESH; ++k) {
                if (last_within[k]) {
                    time_within_s[k] += dt;
                }
            }
        }
        
        // Update current state
        if (p_cex > T(0)) {
            const T rel = std::abs(price_scale / p_cex - T(1));
            for (size_t k = 0; k < N_THRESH - 1; ++k) {
                last_within[k] = (static_cast<double>(rel) <= THRESH_VALUES[k]);
            }
            // Last threshold is dynamic: 1/A (A is stored with 10000 multiplier, so real A = A/10000)
            // inv_A = 1 / (A/10000) = 10000/A
            const double inv_A = (A > T(0)) ? 10000.0 / static_cast<double>(A) : 1.0;
            last_within[N_THRESH - 1] = (static_cast<double>(rel) <= inv_A);
        } else {
            for (size_t k = 0; k < N_THRESH; ++k) {
                last_within[k] = false;
            }
        }
        last_ts_thresh = ts;
        have_thresh = true;
    }
    
    // Sample EMA-smoothed correlation between cex_price and price_scale
    void sample_correlation(uint64_t ts, T price_scale, T p_cex) {
        if (!(p_cex > T(0))) return;
        
        const long double cex_ld = static_cast<long double>(p_cex);
        const long double ps_ld = static_cast<long double>(price_scale);
        
        if (have_corr && ts > last_ts_corr) {
            const double dt = static_cast<double>(ts - last_ts_corr);
            const double alpha = 1.0 - std::exp(-dt / CORR_TAU_S);
            
            // Update EMAs
            ema_cex = (1.0L - alpha) * ema_cex + alpha * cex_ld;
            ema_ps = (1.0L - alpha) * ema_ps + alpha * ps_ld;
            
            // Accumulate correlation stats on smoothed values
            sum_x_dt += ema_cex * dt;
            sum_y_dt += ema_ps * dt;
            sum_xx_dt += ema_cex * ema_cex * dt;
            sum_yy_dt += ema_ps * ema_ps * dt;
            sum_xy_dt += ema_cex * ema_ps * dt;
            sum_corr_dt += dt;
        } else {
            // Initialize EMAs
            ema_cex = cex_ld;
            ema_ps = ps_ld;
        }
        last_ts_corr = ts;
        have_corr = true;
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
    
    double tw_slippage_01(size_t k) const {
        if (k >= N_SIZES || tw_real_dt[k] <= 0.0L) return -1.0;
        return static_cast<double>(tw_real_s01_sum_dt[k] / tw_real_dt[k]);
    }
    
    double tw_slippage_10(size_t k) const {
        if (k >= N_SIZES || tw_real_dt[k] <= 0.0L) return -1.0;
        return static_cast<double>(tw_real_s10_sum_dt[k] / tw_real_dt[k]);
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
    T true_growth_initial{0};
    std::array<T, 2> initial_liq{T(0), T(0)};
    
    // Donation APY (for net calculations)
    T donation_apy{0};
    
    double duration_s() const {
        return (t_end > t_start) ? static_cast<double>(t_end - t_start) : 0.0;
    }
};

} // namespace harness
} // namespace arb
