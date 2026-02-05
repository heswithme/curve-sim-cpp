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
        return apy;
    }
    
    const double exponent = SEC_PER_YEAR / duration_s;
    const double donation_apy = static_cast<double>(r.donation_apy);
    const double donation_freq_s = static_cast<double>(r.donation_frequency);
    double donation_growth = 1.0;
    if (donation_apy > 0.0) {
        if (donation_freq_s > 0.0) {
            const double period_rate = donation_apy * donation_freq_s / SEC_PER_YEAR;
            if (period_rate <= -1.0) {
                donation_growth = -1.0;
            } else {
                donation_growth = std::pow(1.0 + period_rate, duration_s / donation_freq_s);
            }
        } else {
            donation_growth = std::pow(1.0 + donation_apy, duration_s / SEC_PER_YEAR);
        }
    }
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
    
    apy["apy"] = apy_vp;
    apy["apy_net"] = apy_net;
    apy["apy_xcp"] = apy_xcp;
    apy["apy_xcp_net"] = apy_xcp_net;
    
    return apy;
}

// Convert all metrics to summary JSON object (matches original harness format)
template <typename T>
void append_core_metrics(
    json::object& summary,
    const PoolResult<T>& r,
    size_t n_events,
    const TimeWeightedSummary& tw_summary
) {
    const auto& m = r.metrics;
    summary["events"] = static_cast<uint64_t>(n_events);
    summary["trades"] = static_cast<uint64_t>(m.trades);
    summary["total_notional_coin0"] = static_cast<double>(m.notional);
    summary["lp_fee_coin0"] = static_cast<double>(m.lp_fee_coin0);
    summary["avg_pool_fee"] = static_cast<double>(m.avg_pool_fee());
    summary["tw_avg_pool_fee"] = tw_summary.tw_avg_pool_fee;
    summary["arb_pnl_coin0"] = static_cast<double>(m.arb_pnl_coin0);

    const double gross_lvr = static_cast<double>(m.lp_fee_coin0 + m.arb_pnl_coin0);
    summary["fee_capture_rate"] = (gross_lvr > 0.0)
        ? static_cast<double>(m.lp_fee_coin0) / gross_lvr
        : -1.0;

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
}

template <typename T>
void append_slippage_metrics(json::object& summary, const PoolResult<T>& r) {
    const auto& sp = r.slippage_probes;
    summary["tw_real_slippage_1pct"] = sp.tw_slippage(0);
    summary["tw_real_slippage_5pct"] = sp.tw_slippage(1);
    summary["tw_real_slippage_10pct"] = sp.tw_slippage(2);
}

inline void append_price_follow_metrics(
    json::object& summary,
    const TimeWeightedSummary& tw_summary
) {
    summary["avg_rel_price_diff"] = tw_summary.avg_rel_price_diff;
    summary["max_rel_price_diff"] = tw_summary.max_rel_price_diff;
    summary["avg_imbalance"] = tw_summary.avg_imbalance;
}

template <typename T>
void append_tvl_metrics(json::object& summary, const PoolResult<T>& r) {
    summary["t_start"] = r.t_start;
    summary["t_end"] = r.t_end;
    summary["duration_s"] = r.duration_s();
    summary["tvl_coin0_start"] = static_cast<double>(r.tvl_start);

    const T tvl_end = r.balances[0] + r.balances[1] * r.price_scale;
    summary["tvl_coin0_end"] = static_cast<double>(tvl_end);
    summary["tvl_growth"] = static_cast<double>(tvl_end / r.tvl_start);
}

template <typename T>
void append_apy_metrics(json::object& summary, const PoolResult<T>& r) {
    auto apy_metrics = compute_apy_metrics(r);
    for (const auto& kv : apy_metrics) {
        summary[kv.key()] = kv.value();
    }
}

template <typename T>
void append_end_state_metrics(json::object& summary, const PoolResult<T>& r) {
    summary["vp"] = static_cast<double>(r.virtual_price);
    summary["vp_boosted"] = static_cast<double>(r.vp_boosted);
    summary["vpminusone"] = static_cast<double>(r.virtual_price - T(1));
}

template <typename T>
json::object metrics_to_summary(const PoolResult<T>& r, size_t n_events) {
    json::object summary;
    const auto tw_summary = r.tw_metrics.summarize();

    append_core_metrics(summary, r, n_events, tw_summary);
    append_slippage_metrics(summary, r);
    append_price_follow_metrics(summary, tw_summary);
    append_tvl_metrics(summary, r);
    append_apy_metrics(summary, r);
    append_end_state_metrics(summary, r);

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
