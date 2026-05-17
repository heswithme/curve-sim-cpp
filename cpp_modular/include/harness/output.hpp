// JSON output writer for harness results
#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <filesystem>
#include <type_traits>

#include <boost/json.hpp>

#include "core/json_utils.hpp"
#include "harness/npz_writer.hpp"
#include "harness/runner.hpp"

namespace json = boost::json;

namespace arb {
namespace harness {

template <typename T>
double to_wei_double(T v) {
    long double scaled = static_cast<long double>(v) * WAD;
    if (!std::isfinite(scaled)) scaled = 0;
    if (scaled < 0) scaled = 0;
    return static_cast<double>(scaled);
}

struct ApyMetricValues {
    double apy{-1.0};
    double apy_net{-1.0};
    double apy_xcp{-1.0};
    double apy_xcp_net{-1.0};
    double apy_net_gm{-1.0};
};

// Convert pool result's final state to JSON object
template <typename T, typename PoolT>
json::object pool_state_json(const PoolResult<T, PoolT>& r) {
    json::object o;
    o["balances"] = json::array{
        to_str_1e18(r.balances[0]),
        to_str_1e18(r.balances[1])
    };
    o["admin_balances"] = json::array{
        to_str_1e18(r.admin_balances[0]),
        to_str_1e18(r.admin_balances[1])
    };
    // xp = balances scaled with price_scale (matches old harness)
    const T xp1 = r.balances[1] * r.price_scale;
    o["xp"] = json::array{
        to_str_1e18(r.balances[0]),
        to_str_1e18(xp1)
    };
    o["D"] = to_str_1e18(r.D);
    o["virtual_price"] = to_str_1e18(r.virtual_price);
    o["xcp_profit"] = to_str_1e18(r.xcp_profit);
    o["lp_xcp_profit"] = to_str_1e18(r.lp_xcp_profit);
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
template <typename T, typename PoolT>
ApyMetricValues compute_apy_values(const PoolResult<T, PoolT>& r) {
    ApyMetricValues apy{};

    constexpr double SEC_PER_YEAR = 365.0 * 86400.0;
    const double duration_s = r.duration_s();

    if (duration_s <= 0.0 || r.tvl_start <= T(0)) {
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
    apy.apy = (vp_end > 0.0) ? std::pow(vp_end, exponent) - 1.0 : -1.0;
    apy.apy_net = net_apy_from_growth(vp_end);

    // xcp_profit based APY
    const double xcp_end = static_cast<double>((r.xcp_profit + T(1)) / T(2));
    apy.apy_xcp = (xcp_end > 0.0) ? std::pow(xcp_end, exponent) - 1.0 : -1.0;
    apy.apy_xcp_net = net_apy_from_growth(xcp_end);
    apy.apy_net_gm = r.apy_net_gm;

    return apy;
}

template <typename T, typename PoolT>
json::object compute_apy_metrics(const PoolResult<T, PoolT>& r) {
    const auto values = compute_apy_values(r);
    json::object apy;
    apy["apy"] = values.apy;
    apy["apy_net"] = values.apy_net;
    apy["apy_xcp"] = values.apy_xcp;
    apy["apy_xcp_net"] = values.apy_xcp_net;
    apy["apy_net_gm"] = values.apy_net_gm;
    return apy;
}

// Convert all metrics to summary JSON object (matches original harness format)
template <typename T, typename PoolT>
void append_core_metrics(
    json::object& summary,
    const PoolResult<T, PoolT>& r,
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
    summary["min_pool_fee"] = tw_summary.min_pool_fee;
    summary["max_pool_fee"] = tw_summary.max_pool_fee;
    summary["arb_pnl_coin0"] = static_cast<double>(m.arb_pnl_coin0);
    summary["arb_edge_candidates"] = static_cast<uint64_t>(m.arb_edge_candidates);
    summary["arb_invalid_size_rejections"] = static_cast<uint64_t>(m.arb_invalid_size_rejections);
    summary["arb_nonpositive_profit_rejections"] = static_cast<uint64_t>(m.arb_nonpositive_profit_rejections);
    summary["arb_guarded_loss_coin0"] = static_cast<double>(m.arb_guarded_loss_coin0);

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
    const size_t cowswap_processed = m.cowswap_trades + m.cowswap_skipped;
    summary["cowswap_attraction_rate"] = (cowswap_processed > 0)
        ? static_cast<double>(m.cowswap_trades) / static_cast<double>(cowswap_processed)
        : -1.0;
    summary["cowswap_notional_coin0"] = static_cast<double>(m.cowswap_notional_coin0);
    summary["cowswap_lp_fee_coin0"] = static_cast<double>(m.cowswap_lp_fee_coin0);
    const T total_swap_notional = m.notional + m.cowswap_notional_coin0;
    summary["cowswap_vol_rate"] = (total_swap_notional > T(0))
        ? static_cast<double>(m.cowswap_notional_coin0 / total_swap_notional)
        : -1.0;
    summary["pool_exec_ms"] = r.elapsed_ms;
}

template <typename T, typename PoolT>
void append_slippage_metrics(json::object& summary, const PoolResult<T, PoolT>& r) {
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
    summary["max_7d_rel_price_diff"] = tw_summary.max_7d_rel_price_diff;
    summary["min_price_scale"] = tw_summary.min_price_scale;
    summary["max_price_scale"] = tw_summary.max_price_scale;
    summary["avg_imbalance"] = tw_summary.avg_imbalance;
    summary["max_7d_skew"] = tw_summary.max_7d_skew;
}

template <typename T, typename PoolT>
void append_tvl_metrics(json::object& summary, const PoolResult<T, PoolT>& r) {
    summary["t_start"] = r.t_start;
    summary["t_end"] = r.t_end;
    summary["duration_s"] = r.duration_s();
    summary["tvl_coin0_start"] = static_cast<double>(r.tvl_start);

    const T tvl_end = r.balances[0] + r.balances[1] * r.price_scale;
    summary["tvl_coin0_end"] = static_cast<double>(tvl_end);
    summary["tvl_growth"] = static_cast<double>(tvl_end / r.tvl_start);
}

template <typename T, typename PoolT>
void append_apy_metrics(json::object& summary, const PoolResult<T, PoolT>& r) {
    auto apy_metrics = compute_apy_metrics(r);
    for (const auto& kv : apy_metrics) {
        summary[kv.key()] = kv.value();
    }
}

template <typename T, typename PoolT>
void append_end_state_metrics(json::object& summary, const PoolResult<T, PoolT>& r) {
    const T vp = r.virtual_price;
    summary["vp"] = static_cast<double>(vp);
    summary["vp_boosted"] = static_cast<double>(r.vp_boosted);
    summary["vpminusone"] = static_cast<double>(vp - T(1));
}

template <typename T, typename PoolT>
json::object metrics_to_summary(const PoolResult<T, PoolT>& r, size_t n_events) {
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

template <typename T, typename PoolT>
json::object build_run_metadata(
    size_t n_candles,
    size_t n_events,
    size_t n_pools,
    const std::string& data_path,
    const std::string& pools_path,
    const std::string& real_type,
    size_t n_threads,
    const RunConfig<T>& run_cfg,
    size_t n_candles_requested,
    double candle_filter_pct,
    size_t pool_start,
    size_t pool_end,
    bool quiet,
    double candles_read_ms,
    double exec_ms,
    uint64_t total_trades
);

// Output format for the entire run
template <typename T, typename PoolT = T>
json::object build_output_json(
    const std::vector<PoolResult<T, PoolT>>& results,
    size_t n_candles,
    size_t n_events,
    const std::string& data_path,
    const std::string& pools_path,
    const std::string& real_type,
    size_t n_threads,
    const RunConfig<T>& run_cfg,
    size_t n_candles_requested,
    double candle_filter_pct,
    size_t pool_start,
    size_t pool_end,
    bool quiet,
    double candles_read_ms,
    double exec_ms
) {
    uint64_t total_trades = 0;
    for (const auto& r : results) {
        total_trades += static_cast<uint64_t>(r.metrics.trades);
    }
    json::object meta = build_run_metadata<T, PoolT>(
        n_candles,
        n_events,
        results.size(),
        data_path,
        pools_path,
        real_type,
        n_threads,
        run_cfg,
        n_candles_requested,
        candle_filter_pct,
        pool_start,
        pool_end,
        quiet,
        candles_read_ms,
        exec_ms,
        total_trades
    );
    
    // Build runs array
    json::array runs;
    runs.reserve(results.size());
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        
        json::object run;
        run["pool_index"] = static_cast<uint64_t>(r.pool_index);
        
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
template <typename T, typename PoolT = T>
bool write_results_json(
    const std::string& output_path,
    const std::vector<PoolResult<T, PoolT>>& results,
    size_t n_candles,
    size_t n_events,
    const std::string& data_path,
    const std::string& pools_path,
    const std::string& real_type,
    size_t n_threads,
    const RunConfig<T>& run_cfg,
    size_t n_candles_requested,
    double candle_filter_pct,
    size_t pool_start,
    size_t pool_end,
    bool quiet,
    double candles_read_ms,
    double exec_ms
) {
    auto O = build_output_json(
        results, n_candles, n_events, data_path, pools_path, real_type, n_threads,
        run_cfg, n_candles_requested, candle_filter_pct, pool_start, pool_end,
        quiet, candles_read_ms, exec_ms
    );
    
    std::ofstream of(output_path);
    if (!of) {
        return false;
    }
    
    of << json::serialize(O) << '\n';
    return of.good();
}

struct DoubleColumn {
    std::string name;
    std::vector<double> values;
};

struct U64Column {
    std::string name;
    std::vector<uint64_t> values;
};

template <typename T, typename PoolT = T>
json::object build_run_metadata(
    size_t n_candles,
    size_t n_events,
    size_t n_pools,
    const std::string& data_path,
    const std::string& pools_path,
    const std::string& real_type,
    size_t n_threads,
    const RunConfig<T>& run_cfg,
    size_t n_candles_requested,
    double candle_filter_pct,
    size_t pool_start,
    size_t pool_end,
    bool quiet,
    double candles_read_ms,
    double exec_ms,
    uint64_t total_trades
) {
    json::object meta;
    meta["candles_file"] = data_path;
    meta["pool_config_file"] = pools_path;
    meta["real"] = real_type;
    if constexpr (!std::is_same_v<T, PoolT>) {
        meta["harness_real"] = "long double";
        meta["pool_real"] = "uint256";
        meta["numeric_mode"] = "pool_u256";
    }
    meta["n_candles_requested"] = static_cast<uint64_t>(n_candles_requested);
    meta["candles"] = static_cast<uint64_t>(n_candles);
    meta["n_candles_loaded"] = static_cast<uint64_t>(n_candles);
    meta["events"] = static_cast<uint64_t>(n_events);
    meta["n_pools"] = static_cast<uint64_t>(n_pools);
    meta["pool_start"] = static_cast<uint64_t>(pool_start);
    meta["pool_end"] = (pool_end == SIZE_MAX)
        ? json::value(nullptr)
        : json::value(static_cast<uint64_t>(pool_end));
    meta["threads"] = static_cast<uint64_t>(n_threads);
    meta["quiet"] = quiet;
    meta["start_time"] = static_cast<uint64_t>(run_cfg.start_ts);
    meta["disable_slippage_probes"] = !run_cfg.enable_slippage_probes;
    meta["save_actions"] = run_cfg.save_actions;
    meta["min_swap"] = static_cast<double>(run_cfg.min_swap_frac);
    meta["max_swap"] = static_cast<double>(run_cfg.max_swap_frac);
    meta["candle_filter"] = candle_filter_pct;
    meta["dustswapfreq"] = static_cast<uint64_t>(run_cfg.dustswap_freq_s);
    meta["userswapfreq"] = static_cast<uint64_t>(run_cfg.user_swap_freq_s);
    meta["userswapsize"] = static_cast<double>(run_cfg.user_swap_size_frac);
    meta["userswapthresh"] = static_cast<double>(run_cfg.user_swap_thresh);
    meta["cowswap_enabled"] = !run_cfg.cowswap_path.empty();
    meta["cowswap_file"] = run_cfg.cowswap_path;
    meta["cowswap_fee_bps"] = static_cast<double>(run_cfg.cowswap_fee_bps);
    meta["candles_read_ms"] = candles_read_ms;
    meta["exec_ms"] = exec_ms;
    meta["total_trades"] = total_trades;
    return meta;
}

enum class DoubleMetric : size_t {
    TotalNotionalCoin0,
    LpFeeCoin0,
    AvgPoolFee,
    TwAvgPoolFee,
    MinPoolFee,
    MaxPoolFee,
    ArbPnlCoin0,
    ArbGuardedLossCoin0,
    FeeCaptureRate,
    DonationCoin0Total,
    DonationAmountsTotal0,
    DonationAmountsTotal1,
    CowswapAttractionRate,
    CowswapNotionalCoin0,
    CowswapLpFeeCoin0,
    CowswapVolRate,
    PoolExecMs,
    TwRealSlippage1Pct,
    TwRealSlippage5Pct,
    TwRealSlippage10Pct,
    AvgRelPriceDiff,
    MaxRelPriceDiff,
    Max7dRelPriceDiff,
    MinPriceScale,
    MaxPriceScale,
    AvgImbalance,
    Max7dSkew,
    DurationS,
    TvlCoin0Start,
    TvlCoin0End,
    TvlGrowth,
    Apy,
    ApyNet,
    ApyXcp,
    ApyXcpNet,
    Vp,
    VpBoosted,
    VpMinusOne,
    Balances0,
    Balances1,
    AdminBalances0,
    AdminBalances1,
    Xp0,
    Xp1,
    D,
    VirtualPrice,
    XcpProfit,
    LpXcpProfit,
    PriceScale,
    PriceOracle,
    LastPrices,
    TotalSupply,
    DonationShares,
    DonationUnlocked,
    ApyNetGm,
};

enum class U64Metric : size_t {
    PoolIndex,
    Events,
    Trades,
    ArbEdgeCandidates,
    ArbInvalidSizeRejections,
    ArbNonpositiveProfitRejections,
    NRebalances,
    Donations,
    CowswapTrades,
    CowswapSkipped,
    TStart,
    TEnd,
    Timestamp,
};

inline void append_metric(std::vector<DoubleColumn>& cols, DoubleMetric metric, double value) {
    cols[static_cast<size_t>(metric)].values.push_back(value);
}

inline void append_metric(std::vector<U64Column>& cols, U64Metric metric, uint64_t value) {
    cols[static_cast<size_t>(metric)].values.push_back(value);
}

template <typename T, typename PoolT = T>
bool write_results_npz_dir(
    const std::string& output_path,
    const std::vector<PoolResult<T, PoolT>>& results,
    size_t n_candles,
    size_t n_events,
    const std::string& data_path,
    const std::string& pools_path,
    const std::string& real_type,
    size_t n_threads,
    const RunConfig<T>& run_cfg,
    size_t n_candles_requested,
    double candle_filter_pct,
    size_t pool_start,
    size_t pool_end,
    bool quiet,
    double candles_read_ms,
    double exec_ms
) {
    namespace fs = std::filesystem;
    const fs::path out_dir(output_path);
    if (fs::exists(out_dir) && !fs::is_directory(out_dir)) {
        return false;
    }
    fs::create_directories(out_dir);

    const size_t n = results.size();
    std::vector<DoubleColumn> dcols = {
        {"total_notional_coin0", {}},
        {"lp_fee_coin0", {}},
        {"avg_pool_fee", {}},
        {"tw_avg_pool_fee", {}},
        {"min_pool_fee", {}},
        {"max_pool_fee", {}},
        {"arb_pnl_coin0", {}},
        {"arb_guarded_loss_coin0", {}},
        {"fee_capture_rate", {}},
        {"donation_coin0_total", {}},
        {"donation_amounts_total_0", {}},
        {"donation_amounts_total_1", {}},
        {"cowswap_attraction_rate", {}},
        {"cowswap_notional_coin0", {}},
        {"cowswap_lp_fee_coin0", {}},
        {"cowswap_vol_rate", {}},
        {"pool_exec_ms", {}},
        {"tw_real_slippage_1pct", {}},
        {"tw_real_slippage_5pct", {}},
        {"tw_real_slippage_10pct", {}},
        {"avg_rel_price_diff", {}},
        {"max_rel_price_diff", {}},
        {"max_7d_rel_price_diff", {}},
        {"min_price_scale", {}},
        {"max_price_scale", {}},
        {"avg_imbalance", {}},
        {"max_7d_skew", {}},
        {"duration_s", {}},
        {"tvl_coin0_start", {}},
        {"tvl_coin0_end", {}},
        {"tvl_growth", {}},
        {"apy", {}},
        {"apy_net", {}},
        {"apy_xcp", {}},
        {"apy_xcp_net", {}},
        {"vp", {}},
        {"vp_boosted", {}},
        {"vpminusone", {}},
        {"balances_0", {}},
        {"balances_1", {}},
        {"admin_balances_0", {}},
        {"admin_balances_1", {}},
        {"xp_0", {}},
        {"xp_1", {}},
        {"D", {}},
        {"virtual_price", {}},
        {"xcp_profit", {}},
        {"lp_xcp_profit", {}},
        {"price_scale", {}},
        {"price_oracle", {}},
        {"last_prices", {}},
        {"totalSupply", {}},
        {"donation_shares", {}},
        {"donation_unlocked", {}},
        {"apy_net_gm", {}},
    };
    std::vector<U64Column> ucols = {
        {"pool_index", {}},
        {"events", {}},
        {"trades", {}},
        {"arb_edge_candidates", {}},
        {"arb_invalid_size_rejections", {}},
        {"arb_nonpositive_profit_rejections", {}},
        {"n_rebalances", {}},
        {"donations", {}},
        {"cowswap_trades", {}},
        {"cowswap_skipped", {}},
        {"t_start", {}},
        {"t_end", {}},
        {"timestamp", {}},
    };
    std::vector<uint8_t> success;

    for (auto& c : dcols) c.values.reserve(n);
    for (auto& c : ucols) c.values.reserve(n);
    success.reserve(n);

    auto d = [&](DoubleMetric metric, double value) { append_metric(dcols, metric, value); };
    auto u = [&](U64Metric metric, uint64_t value) { append_metric(ucols, metric, value); };

    uint64_t total_trades = 0;
    json::object errors;
    for (size_t i = 0; i < n; ++i) {
        const auto& r = results[i];
        const auto tw = r.tw_metrics.summarize();
        const auto apy = compute_apy_values(r);
        const auto& m = r.metrics;
        const double gross_lvr = static_cast<double>(m.lp_fee_coin0 + m.arb_pnl_coin0);
        const T total_swap_notional = m.notional + m.cowswap_notional_coin0;
        const T tvl_end = r.balances[0] + r.balances[1] * r.price_scale;
        const T vp = r.virtual_price;

        total_trades += static_cast<uint64_t>(m.trades);
        if (!r.success) {
            errors[std::to_string(r.pool_index)] = r.error_msg;
        }

        d(DoubleMetric::TotalNotionalCoin0, static_cast<double>(m.notional));
        d(DoubleMetric::LpFeeCoin0, static_cast<double>(m.lp_fee_coin0));
        d(DoubleMetric::AvgPoolFee, static_cast<double>(m.avg_pool_fee()));
        d(DoubleMetric::TwAvgPoolFee, tw.tw_avg_pool_fee);
        d(DoubleMetric::MinPoolFee, tw.min_pool_fee);
        d(DoubleMetric::MaxPoolFee, tw.max_pool_fee);
        d(DoubleMetric::ArbPnlCoin0, static_cast<double>(m.arb_pnl_coin0));
        d(DoubleMetric::ArbGuardedLossCoin0, static_cast<double>(m.arb_guarded_loss_coin0));
        d(DoubleMetric::FeeCaptureRate, gross_lvr > 0.0 ? static_cast<double>(m.lp_fee_coin0) / gross_lvr : -1.0);
        d(DoubleMetric::DonationCoin0Total, static_cast<double>(m.donation_coin0_total));
        d(DoubleMetric::DonationAmountsTotal0, static_cast<double>(m.donation_amounts_total[0]));
        d(DoubleMetric::DonationAmountsTotal1, static_cast<double>(m.donation_amounts_total[1]));
        const size_t cowswap_processed = m.cowswap_trades + m.cowswap_skipped;
        d(DoubleMetric::CowswapAttractionRate, cowswap_processed > 0
            ? static_cast<double>(m.cowswap_trades) / static_cast<double>(cowswap_processed)
            : -1.0);
        d(DoubleMetric::CowswapNotionalCoin0, static_cast<double>(m.cowswap_notional_coin0));
        d(DoubleMetric::CowswapLpFeeCoin0, static_cast<double>(m.cowswap_lp_fee_coin0));
        d(DoubleMetric::CowswapVolRate, total_swap_notional > T(0)
            ? static_cast<double>(m.cowswap_notional_coin0 / total_swap_notional)
            : -1.0);
        d(DoubleMetric::PoolExecMs, r.elapsed_ms);
        d(DoubleMetric::TwRealSlippage1Pct, r.slippage_probes.tw_slippage(0));
        d(DoubleMetric::TwRealSlippage5Pct, r.slippage_probes.tw_slippage(1));
        d(DoubleMetric::TwRealSlippage10Pct, r.slippage_probes.tw_slippage(2));
        d(DoubleMetric::AvgRelPriceDiff, tw.avg_rel_price_diff);
        d(DoubleMetric::MaxRelPriceDiff, tw.max_rel_price_diff);
        d(DoubleMetric::Max7dRelPriceDiff, tw.max_7d_rel_price_diff);
        d(DoubleMetric::MinPriceScale, tw.min_price_scale);
        d(DoubleMetric::MaxPriceScale, tw.max_price_scale);
        d(DoubleMetric::AvgImbalance, tw.avg_imbalance);
        d(DoubleMetric::Max7dSkew, tw.max_7d_skew);
        d(DoubleMetric::DurationS, r.duration_s());
        d(DoubleMetric::TvlCoin0Start, static_cast<double>(r.tvl_start));
        d(DoubleMetric::TvlCoin0End, static_cast<double>(tvl_end));
        d(DoubleMetric::TvlGrowth, static_cast<double>(tvl_end / r.tvl_start));
        d(DoubleMetric::Apy, apy.apy);
        d(DoubleMetric::ApyNet, apy.apy_net);
        d(DoubleMetric::ApyXcp, apy.apy_xcp);
        d(DoubleMetric::ApyXcpNet, apy.apy_xcp_net);
        d(DoubleMetric::Vp, static_cast<double>(vp));
        d(DoubleMetric::VpBoosted, static_cast<double>(r.vp_boosted));
        d(DoubleMetric::VpMinusOne, static_cast<double>(vp - T(1)));
        d(DoubleMetric::Balances0, to_wei_double(r.balances[0]));
        d(DoubleMetric::Balances1, to_wei_double(r.balances[1]));
        d(DoubleMetric::AdminBalances0, to_wei_double(r.admin_balances[0]));
        d(DoubleMetric::AdminBalances1, to_wei_double(r.admin_balances[1]));
        d(DoubleMetric::Xp0, to_wei_double(r.balances[0]));
        d(DoubleMetric::Xp1, to_wei_double(r.balances[1] * r.price_scale));
        d(DoubleMetric::D, to_wei_double(r.D));
        d(DoubleMetric::VirtualPrice, to_wei_double(r.virtual_price));
        d(DoubleMetric::XcpProfit, to_wei_double(r.xcp_profit));
        d(DoubleMetric::LpXcpProfit, to_wei_double(r.lp_xcp_profit));
        d(DoubleMetric::PriceScale, to_wei_double(r.price_scale));
        d(DoubleMetric::PriceOracle, to_wei_double(r.price_oracle));
        d(DoubleMetric::LastPrices, to_wei_double(r.last_prices));
        d(DoubleMetric::TotalSupply, to_wei_double(r.totalSupply));
        d(DoubleMetric::DonationShares, to_wei_double(r.donation_shares));
        d(DoubleMetric::DonationUnlocked, to_wei_double(r.donation_unlocked));
        d(DoubleMetric::ApyNetGm, apy.apy_net_gm);

        u(U64Metric::PoolIndex, static_cast<uint64_t>(r.pool_index));
        u(U64Metric::Events, static_cast<uint64_t>(n_events));
        u(U64Metric::Trades, static_cast<uint64_t>(m.trades));
        u(U64Metric::ArbEdgeCandidates, static_cast<uint64_t>(m.arb_edge_candidates));
        u(U64Metric::ArbInvalidSizeRejections, static_cast<uint64_t>(m.arb_invalid_size_rejections));
        u(U64Metric::ArbNonpositiveProfitRejections, static_cast<uint64_t>(m.arb_nonpositive_profit_rejections));
        u(U64Metric::NRebalances, static_cast<uint64_t>(m.n_rebalances));
        u(U64Metric::Donations, static_cast<uint64_t>(m.donations));
        u(U64Metric::CowswapTrades, static_cast<uint64_t>(m.cowswap_trades));
        u(U64Metric::CowswapSkipped, static_cast<uint64_t>(m.cowswap_skipped));
        u(U64Metric::TStart, r.t_start);
        u(U64Metric::TEnd, r.t_end);
        u(U64Metric::Timestamp, r.timestamp);
        success.push_back(r.success ? 1U : 0U);
    }

    try {
        StoredNpzWriter writer((out_dir / "metrics.npz").string());
        for (const auto& c : dcols) writer.add_f64(c.name, c.values);
        for (const auto& c : ucols) writer.add_u64(c.name, c.values);
        writer.add_u8("success", success);
        writer.close();

        json::array metrics_schema;
        for (const auto& c : dcols) {
            metrics_schema.push_back(json::object{{"name", c.name}, {"dtype", "float64"}});
        }
        for (const auto& c : ucols) {
            metrics_schema.push_back(json::object{{"name", c.name}, {"dtype", "uint64"}});
        }
        metrics_schema.push_back(json::object{{"name", "success"}, {"dtype", "uint8"}});

        json::object meta = build_run_metadata<T, PoolT>(
            n_candles,
            n_events,
            n,
            data_path,
            pools_path,
            real_type,
            n_threads,
            run_cfg,
            n_candles_requested,
            candle_filter_pct,
            pool_start,
            pool_end,
            quiet,
            candles_read_ms,
            exec_ms,
            total_trades
        );

        json::object manifest;
        manifest["format"] = "arb_npz_v1";
        manifest["metrics_file"] = "metrics.npz";
        manifest["n_pools"] = static_cast<uint64_t>(n);
        manifest["pool_start"] = static_cast<uint64_t>(pool_start);
        manifest["pool_end"] = (pool_end == SIZE_MAX)
            ? json::value(nullptr)
            : json::value(static_cast<uint64_t>(pool_end));
        manifest["metadata"] = std::move(meta);
        manifest["metrics_schema"] = std::move(metrics_schema);

        std::ofstream mf(out_dir / "manifest.json");
        if (!mf) return false;
        mf << json::serialize(manifest) << '\n';

        if (!errors.empty()) {
            std::ofstream ef(out_dir / "errors.json");
            if (!ef) return false;
            ef << json::serialize(errors) << '\n';
        } else {
            std::error_code ec;
            fs::remove(out_dir / "errors.json", ec);
        }
    } catch (const std::exception&) {
        return false;
    }

    return true;
}

} // namespace harness
} // namespace arb
