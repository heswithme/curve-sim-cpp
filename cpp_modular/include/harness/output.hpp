// JSON output writer for harness results
#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <boost/json.hpp>

#include "core/json_utils.hpp"
#include "harness/runner.hpp"
#include "pools/twocrypto_fx/helpers.hpp"

namespace json = boost::json;

namespace arb {
namespace harness {

// Convert pool result's final state to JSON object
template <typename T>
json::object pool_state_json(const PoolResult<T>& r) {
    json::object o;
    o["balances"] = json::array{to_str_1e18(r.balances[0]), to_str_1e18(r.balances[1])};
    // xp = balances scaled with price_scale (matches old harness)
    o["xp"] = json::array{
        to_str_1e18(r.balances[0]),  // precisions[0] = 1 for floating types
        to_str_1e18(r.balances[1] * r.price_scale)  // precisions[1] = 1 for floating types
    };
    o["D"] = to_str_1e18(r.D);
    o["virtual_price"] = to_str_1e18(r.virtual_price);
    o["xcp_profit"] = to_str_1e18(r.xcp_profit);
    o["price_scale"] = to_str_1e18(r.price_scale);
    o["price_oracle"] = to_str_1e18(r.price_oracle);
    o["last_prices"] = to_str_1e18(r.last_prices);
    o["totalSupply"] = to_str_1e18(r.totalSupply);
    o["donation_shares"] = to_str_1e18(r.donation_shares);
    o["donation_unlocked"] = to_str_1e18(r.donation_unlocked);
    o["timestamp"] = r.timestamp;
    return o;
}

// Compute end-state APY metrics from PoolResult
template <typename T>
json::object compute_apy_metrics(const PoolResult<T>& r) {
    json::object apy;
    
    constexpr double SEC_PER_YEAR = 365.0 * 86400.0;
    const double duration_s = r.duration_s();
    
    if (duration_s <= 0.0 || r.tvl_start <= T(0)) {
        // Can't compute APYs
        apy["apy"] = -1.0;
        apy["apy_net"] = -1.0;
        apy["apy_xcp"] = -1.0;
        apy["apy_xcp_net"] = -1.0;
        apy["apy_coin0"] = -1.0;
        apy["apy_coin0_boost"] = -1.0;
        apy["apy_coin0_raw"] = -1.0;
        apy["apy_coin0_boost_raw"] = -1.0;
        apy["apy_geom_mean"] = -1.0;
        apy["apy_geom_mean_net"] = -1.0;
        return apy;
    }
    
    const double exponent = SEC_PER_YEAR / duration_s;
    const double donation_apy = static_cast<double>(r.donation_apy);
    const double donation_growth = (donation_apy > -1.0)
        ? std::pow(1.0 + donation_apy, duration_s / SEC_PER_YEAR)
        : -1.0;
    auto net_apy_from_growth = [&](double gross_growth) -> double {
        if (!(gross_growth > 0.0) || !(donation_growth > 0.0)) return -1.0;
        const double net_growth = gross_growth / donation_growth;
        return (net_growth > 0.0) ? std::pow(net_growth, exponent) - 1.0 : -1.0;
    };
    
    // Virtual price based APY
    const double vp_end = static_cast<double>(r.virtual_price);
    double apy_vp = (vp_end > 0.0) ? std::pow(vp_end, exponent) - 1.0 : -1.0;
    
    double apy_net = net_apy_from_growth(vp_end);
    // xcp_profit based APY
    const double xcp_end = static_cast<double>((r.xcp_profit + T(1)) / T(2));
    double apy_xcp = (xcp_end > 0.0) ? std::pow(xcp_end, exponent) - 1.0 : -1.0;
    double apy_xcp_net = net_apy_from_growth(xcp_end);
    
    // TVL based APYs
    const T tvl_end = r.balances[0] + r.balances[1] * r.price_scale;
    const T v_hold_end = r.initial_liq[0] + r.initial_liq[1] * r.price_scale;
    
    // Raw (non-baseline) APYs
    double apy_coin0_raw = (tvl_end > T(0)) 
        ? std::pow(static_cast<double>(tvl_end / r.tvl_start), exponent) - 1.0 
        : -1.0;
    
    const double tvl_end_adj_raw = static_cast<double>(tvl_end) - static_cast<double>(r.metrics.donation_coin0_total);
    double apy_coin0_boost_raw = (tvl_end_adj_raw > 0.0)
        ? std::pow(tvl_end_adj_raw / static_cast<double>(r.tvl_start), exponent) - 1.0
        : -1.0;
    
    // Baseline-subtracted (excess) APYs: compare to HODL valued at end price
    double apy_coin0 = -1.0;
    double apy_coin0_boost = -1.0;
    if (v_hold_end > T(0)) {
        apy_coin0 = std::pow(static_cast<double>(tvl_end / v_hold_end), exponent) - 1.0;
        const double tvl_end_adj = static_cast<double>(tvl_end) - static_cast<double>(r.metrics.donation_coin0_total);
        apy_coin0_boost = (tvl_end_adj > 0.0)
            ? std::pow(tvl_end_adj / static_cast<double>(v_hold_end), exponent) - 1.0
            : -1.0;
    }
    
    // True growth based APY
    const T true_growth_end = pools::twocrypto_fx::true_growth_from_balances<T>(
        r.balances, r.price_scale);
    double apy_geom_mean = -1.0;
    double apy_geom_mean_net = -1.0;
    if (r.true_growth_initial > T(0) && true_growth_end > T(0)) {
        const double ratio = static_cast<double>(true_growth_end / r.true_growth_initial);
        apy_geom_mean = std::pow(ratio, exponent) - 1.0;
        apy_geom_mean_net = net_apy_from_growth(ratio);
    }
    
    apy["apy"] = apy_vp;
    apy["apy_net"] = apy_net;
    apy["apy_xcp"] = apy_xcp;
    apy["apy_xcp_net"] = apy_xcp_net;
    apy["apy_coin0"] = apy_coin0;
    apy["apy_coin0_boost"] = apy_coin0_boost;
    apy["apy_coin0_raw"] = apy_coin0_raw;
    apy["apy_coin0_boost_raw"] = apy_coin0_boost_raw;
    apy["apy_geom_mean"] = apy_geom_mean;
    apy["apy_geom_mean_net"] = apy_geom_mean_net;
    
    return apy;
}

// Convert all metrics to summary JSON object (matches original harness format)
template <typename T>
json::object metrics_to_summary(const PoolResult<T>& r, size_t n_events) {
    json::object summary;
    const auto& m = r.metrics;
    const auto& tw = r.tw_metrics;
    const auto& sp = r.slippage_probes;
    
    // Core metrics
    summary["events"] = static_cast<uint64_t>(n_events);
    summary["trades"] = static_cast<uint64_t>(m.trades);
    summary["total_notional_coin0"] = static_cast<double>(m.notional);
    summary["lp_fee_coin0"] = static_cast<double>(m.lp_fee_coin0);
    summary["avg_pool_fee"] = static_cast<double>(m.avg_pool_fee());
    summary["tw_avg_pool_fee"] = tw.tw_avg_pool_fee();
    summary["arb_pnl_coin0"] = static_cast<double>(m.arb_pnl_coin0);
    
    // Fee capture rate: fraction of gross LVR recovered by fees
    // fee_capture_rate = lp_fee / (lp_fee + arb_pnl), -1 if no arb activity
    {
        const double gross_lvr = static_cast<double>(m.lp_fee_coin0 + m.arb_pnl_coin0);
        summary["fee_capture_rate"] = (gross_lvr > 0.0) 
            ? static_cast<double>(m.lp_fee_coin0) / gross_lvr 
            : -1.0;
    }
    
    summary["n_rebalances"] = static_cast<uint64_t>(m.n_rebalances);
    summary["donations"] = static_cast<uint64_t>(m.donations);
    summary["donation_coin0_total"] = static_cast<double>(m.donation_coin0_total);
    summary["donation_amounts_total"] = json::array{
        static_cast<double>(m.donation_amounts_total[0]),
        static_cast<double>(m.donation_amounts_total[1])
    };
    summary["cowswap_trades"] = static_cast<uint64_t>(m.cowswap_trades);
    summary["cowswap_skipped"] = static_cast<uint64_t>(m.cowswap_skipped);
    summary["cowswap_notional_coin0"] = static_cast<double>(m.cowswap_notional_coin0);
    summary["cowswap_lp_fee_coin0"] = static_cast<double>(m.cowswap_lp_fee_coin0);
    summary["pool_exec_ms"] = r.elapsed_ms;
    
    // Time-weighted latent slippage/liquidity density (normalized vs xy=k)
    summary["tw_slippage"] = tw.tw_slippage();
    summary["tw_liq_density"] = tw.tw_liq_density();
    
    // Real slippage probes
    summary["tw_real_slippage_1pct_01"] = sp.tw_slippage_01(0);
    summary["tw_real_slippage_1pct_10"] = sp.tw_slippage_10(0);
    summary["tw_real_slippage_5pct_01"] = sp.tw_slippage_01(1);
    summary["tw_real_slippage_5pct_10"] = sp.tw_slippage_10(1);
    summary["tw_real_slippage_10pct_01"] = sp.tw_slippage_01(2);
    summary["tw_real_slippage_10pct_10"] = sp.tw_slippage_10(2);
    summary["tw_real_slippage_1pct"] = sp.tw_slippage(0);
    summary["tw_real_slippage_5pct"] = sp.tw_slippage(1);
    summary["tw_real_slippage_10pct"] = sp.tw_slippage(2);
    
    // Price follow metrics
    summary["avg_rel_price_diff"] = tw.avg_rel_price_diff();
    summary["max_rel_price_diff"] = tw.max_rel_price_diff();
    summary["cex_follow_time_frac"] = tw.cex_follow_time_frac(r.duration_s());
    
    // Multi-threshold price tracking: fraction of time within X% of CEX
    const double frac_3 = tw.pct_within(0, r.duration_s());
    const double frac_5 = tw.pct_within(1, r.duration_s());
    const double frac_10 = tw.pct_within(2, r.duration_s());
    const double frac_20 = tw.pct_within(3, r.duration_s());
    const double frac_30 = tw.pct_within(4, r.duration_s());
    const double frac_inv_A = tw.pct_within(5, r.duration_s());
    
    summary["frac_within_3pct"] = frac_3;
    summary["frac_within_5pct"] = frac_5;
    summary["frac_within_10pct"] = frac_10;
    summary["frac_within_20pct"] = frac_20;
    summary["frac_within_30pct"] = frac_30;
    summary["frac_within_inv_A"] = frac_inv_A;
    
    // EMA-smoothed price correlation (1hr window)
    const double corr = tw.price_correlation();
    summary["price_correlation"] = corr;
    
    // Timestamps and duration
    summary["t_start"] = r.t_start;
    summary["t_end"] = r.t_end;
    summary["duration_s"] = r.duration_s();
    summary["tvl_coin0_start"] = static_cast<double>(r.tvl_start);
    
    // End-state TVL
    const T tvl_end = r.balances[0] + r.balances[1] * r.price_scale;
    summary["tvl_coin0_end"] = static_cast<double>(tvl_end);
    
    // TVL growth
    summary["tvl_growth"] = static_cast<double>(tvl_end / r.tvl_start);
    
    // Baseline HODL value at end price
    const T v_hold_end = r.initial_liq[0] + r.initial_liq[1] * r.price_scale;
    summary["baseline_hold_end_coin0"] = static_cast<double>(v_hold_end);
    
    // Compute and merge APY metrics
    auto apy_metrics = compute_apy_metrics(r);
    for (const auto& kv : apy_metrics) {
        summary[kv.key()] = kv.value();
    }
    
    // Masked APY metrics: apy_net * frac_within_X
    const double apy_net_val = apy_metrics["apy_net"].as_double();
    summary["apy_mask_3"] = apy_net_val * frac_3;
    summary["apy_mask_5"] = apy_net_val * frac_5;
    summary["apy_mask_10"] = apy_net_val * frac_10;
    summary["apy_mask_20"] = apy_net_val * frac_20;
    summary["apy_mask_30"] = apy_net_val * frac_30;
    summary["apy_mask_inv_A"] = apy_net_val * frac_inv_A;
    
    // Correlation-adjusted APY: full credit at corr >= 0.75, linear penalty below
    const double corr_factor = (corr > 0.0) ? std::min(1.0, corr / 0.75) : 0.0;
    summary["apy_corr"] = apy_net_val * corr_factor;
    
    // Additional end-state metrics
    summary["vp"] = static_cast<double>(r.virtual_price);
    summary["vp_boosted"] = static_cast<double>(r.vp_boosted);
    summary["vpminusone"] = static_cast<double>(r.virtual_price - T(1));
    
    // True growth
    const T true_growth_end = pools::twocrypto_fx::true_growth_from_balances<T>(
        r.balances, r.price_scale);
    if (r.true_growth_initial > T(0)) {
        summary["true_growth"] = static_cast<double>(true_growth_end / r.true_growth_initial);
    } else {
        summary["true_growth"] = -1.0;
    }
    
    return summary;
}

// Output format for the entire run
template <typename T>
json::object build_output_json(
    const std::vector<PoolResult<T>>& results,
    size_t n_events,
    const std::string& data_path,
    size_t n_threads,
    double candles_read_ms,
    double exec_ms
) {
    // Metadata
    json::object meta;
    meta["candles_file"] = data_path;
    meta["events"] = static_cast<uint64_t>(n_events);
    meta["threads"] = static_cast<uint64_t>(n_threads);
    meta["candles_read_ms"] = candles_read_ms;
    meta["exec_ms"] = exec_ms;
    
    // Build runs array
    json::array runs;
    runs.reserve(results.size());
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        
        json::object run;
        
        // Result summary (includes all metrics now)
        run["result"] = metrics_to_summary(r, n_events);
        
        // Params block (echoes back original pool/costs JSON)
        json::object params;
        params["pool"] = r.echo_pool;
        if (!r.echo_costs.empty()) {
            params["costs"] = r.echo_costs;
        }
        run["params"] = params;
        
        // Final state
        run["final_state"] = pool_state_json(r);
        
        // Success flag
        run["success"] = r.success;
        if (!r.success) {
            run["error"] = r.error_msg;
        }
        
        // Actions array (only if save_actions was enabled and we have actions)
        if (!r.actions.empty()) {
            run["actions"] = actions_to_json(r.actions);
        }
        
        runs.push_back(std::move(run));
    }
    
    // Build output object
    json::object O;
    O["metadata"] = meta;
    O["runs"] = runs;
    
    return O;
}

// Write results to JSON file
template <typename T>
bool write_results_json(
    const std::string& output_path,
    const std::vector<PoolResult<T>>& results,
    size_t n_events,
    const std::string& data_path,
    size_t n_threads,
    double candles_read_ms,
    double exec_ms
) {
    auto O = build_output_json(
        results, n_events, data_path, n_threads,
        candles_read_ms, exec_ms
    );
    
    std::ofstream of(output_path);
    if (!of) {
        return false;
    }
    
    of << json::serialize(O) << '\n';
    return of.good();
}

} // namespace harness
} // namespace arb
