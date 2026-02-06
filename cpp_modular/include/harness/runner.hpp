// Pool runner - single pool execution and parallel multi-pool processing
#pragma once

#include <array>
#include <atomic>
#include <future>
#include <string>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <iomanip>

#include "core/common.hpp"
#include "events/types.hpp"
#include "harness/metrics.hpp"
#include "harness/actions.hpp"
#include "harness/detailed_output.hpp"
#include "harness/donation.hpp"
#include "harness/idle_tick.hpp"
#include "harness/user_swap.hpp"
#include "harness/event_loop.hpp"
#include "pools/config.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"
#include "pools/twocrypto_fx/helpers.hpp"
#include "trading/costs.hpp"
#include "trading/cowswap_trader.hpp"

namespace arb {
namespace harness {

// Result from running a single pool - includes all metrics
template <typename T>
struct PoolResult {
    std::string tag;
    
    // Core trading metrics
    Metrics<T> metrics{};
    
    // Time-weighted metrics
    TimeWeightedMetrics<T> tw_metrics{};
    
    // Slippage probes
    SlippageProbes<T> slippage_probes{};
    
    // Start/end timestamps
    uint64_t t_start{0};
    uint64_t t_end{0};
    
    // Initial state (for APY calculations)
    T tvl_start{0};
    T donation_apy{0};
    T donation_frequency{0};
    
    // Final pool state
    std::array<T, 2> balances{T(0), T(0)};
    T D{0};
    T totalSupply{0};
    T price_scale{0};
    T price_oracle{0};
    T virtual_price{0};
    T xcp_profit{0};
    T vp_boosted{0};
    T donation_shares{0};
    T donation_unlocked{0};
    T last_prices{0};
    uint64_t timestamp{0};
    
    // Echo back original JSON for params block
    boost::json::object echo_pool{};
    boost::json::object echo_costs{};
    
    // Actions (only populated if save_actions=true)
    std::vector<Action<T>> actions{};
    
    // Detailed per-candle entries (only populated if detailed_log=true)
    std::vector<DetailedEntry<T>> detailed_entries{};
    
    // Timing
    double elapsed_ms{0};
    
    // Success flag
    bool success{false};
    std::string error_msg;
    
    // Computed helpers
    double duration_s() const {
        return (t_end > t_start) ? static_cast<double>(t_end - t_start) : 0.0;
    }
};

// Configuration for running pools
template <typename T>
struct RunConfig {
    T min_swap_frac{T(1e-6)};
    T max_swap_frac{T(1.0)};
    uint64_t dustswap_freq_s{3600};
    uint64_t user_swap_freq_s{0};
    T user_swap_size_frac{T(0.01)};
    T user_swap_thresh{T(0.05)};
    
    // Action recording
    bool save_actions{false};
    
    // Detailed per-event logging
    bool detailed_log{false};
    size_t detailed_interval{1};  // log every N-th event (1 = all)

    // Slippage probe sampling
    bool enable_slippage_probes{true};
    
    // Cowswap organic trades
    std::string cowswap_path;  // path to cowswap trades CSV (empty = disabled)
    T cowswap_fee_bps{T(0)};   // fee in bps to beat historical execution
};

// Run a single pool configuration and return results
template <typename T>
PoolResult<T> run_single_pool(
    const pools::PoolInit<T>& pool_init,
    const trading::Costs<T>& costs,
    const std::vector<Event>& events,
    const RunConfig<T>& cfg,
    const std::vector<trading::CowswapTrade>* cowswap_trades = nullptr,
    const std::vector<Candle>* candles = nullptr
) {
    using Pool = pools::twocrypto_fx::TwoCryptoPool<T>;
    
    PoolResult<T> result;
    result.tag = pool_init.tag;
    result.echo_pool = pool_init.echo_pool;
    result.echo_costs = pool_init.echo_costs;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    try {
        // Determine initial price
        T initial_price = pool_init.initial_price;
        if (initial_price <= T(0) && !events.empty()) {
            initial_price = static_cast<T>(events.front().p_cex);
        }
        if (initial_price <= T(0)) {
            initial_price = T(1);  // fallback
        }
        
        // Create pool
        Pool pool(
            pool_init.precisions,
            pool_init.A,
            pool_init.gamma,
            pool_init.mid_fee,
            pool_init.out_fee,
            pool_init.fee_gamma,
            pool_init.allowed_extra_profit,
            pool_init.adjustment_step,
            pool_init.ma_time,
            initial_price
        );
        
        // Set initial timestamp
        uint64_t init_ts = pool_init.start_ts;
        if (init_ts == 0 && !events.empty()) {
            init_ts = events.front().ts;
        }
        if (init_ts == 0) {
            init_ts = 1700000000ULL;
        }
        pool.set_block_timestamp(init_ts);
        
        // Add initial liquidity
        T liq0 = pool_init.initial_liq[0];
        T liq1 = pool_init.initial_liq[1];
        if (liq0 <= T(0) || liq1 <= T(0)) {
            liq0 = T(1000000.0);
            liq1 = liq0 / initial_price;
        }
        pool.add_liquidity({liq0, liq1}, T(0));
        
        // Initialize donation config (also locks in base TVL for no-compounding)
        DonationCfg<T> dcfg{};
        if (pool_init.donation_apy > T(0) && pool_init.donation_frequency > T(0) && !events.empty()) {
            dcfg.init(
                pool_init.donation_apy,
                pool_init.donation_frequency,
                pool_init.donation_coins_ratio,
                init_ts,
                pool
            );
        }
        
        // Initialize idle tick config
        IdleTickCfg<T> icfg{};
        icfg.freq_s = cfg.dustswap_freq_s;
        
        // Initialize user swap config
        UserSwapCfg<T> ucfg{};
        ucfg.freq_s = cfg.user_swap_freq_s;
        ucfg.size_frac = cfg.user_swap_size_frac;
        ucfg.thresh = cfg.user_swap_thresh;
        if (ucfg.enabled() && !events.empty()) {
            ucfg.init(init_ts);
        }
        
        // Cowswap trader: create from shared trades pointer, initialize at first event timestamp
        trading::CowswapTrader<T> cowswap_trader;
        trading::CowswapTrader<T>* cowswap_ptr = nullptr;
        if (cowswap_trades && !cowswap_trades->empty() && !events.empty()) {
            cowswap_trader = trading::CowswapTrader<T>(cowswap_trades, cfg.cowswap_fee_bps);
            cowswap_trader.init_at(events.front().ts);
            cowswap_ptr = &cowswap_trader;
        }
        
        // Run event loop
        auto loop_result = run_event_loop(
            pool, events, costs, dcfg, icfg, ucfg,
            cfg.min_swap_frac, cfg.max_swap_frac, 0,
            cfg.enable_slippage_probes,
            cfg.save_actions, cfg.detailed_log, cfg.detailed_interval,
            cowswap_ptr,
            candles
        );
        
        // Copy metrics from event loop result
        result.metrics = loop_result.metrics;
        result.tw_metrics = loop_result.tw_metrics;
        result.slippage_probes = loop_result.slippage_probes;
        result.t_start = loop_result.t_start;
        result.t_end = loop_result.t_end;
        result.tvl_start = loop_result.tvl_start;
        result.donation_apy = loop_result.donation_apy;
        result.donation_frequency = pool_init.donation_frequency;
        
        // Copy actions if recorded
        if (cfg.save_actions) {
            result.actions = std::move(loop_result.actions);
        }
        
        // Copy detailed entries if recorded
        if (cfg.detailed_log) {
            result.detailed_entries = std::move(loop_result.detailed_entries);
        }
        
        // Capture final pool state
        result.balances[0] = pool.balances[0];
        result.balances[1] = pool.balances[1];
        result.D = pool.D;
        result.totalSupply = pool.totalSupply;
        result.price_scale = pool.cached_price_scale;
        result.price_oracle = pool.cached_price_oracle;
        result.virtual_price = pool.get_virtual_price();
        result.xcp_profit = pool.xcp_profit;
        result.vp_boosted = pool.get_vp_boosted();
        result.donation_shares = pool.donation_shares;
        result.donation_unlocked = pool.donation_unlocked();
        result.last_prices = pool.last_prices;
        result.timestamp = pool.block_timestamp;
        result.success = true;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_msg = e.what();
    } catch (...) {
        result.success = false;
        result.error_msg = "Unknown error";
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    return result;
}

// Helper to format seconds as HH:MM:SS or MM:SS
inline std::string format_duration(double seconds) {
    int total_s = static_cast<int>(seconds + 0.5);
    int h = total_s / 3600;
    int m = (total_s % 3600) / 60;
    int s = total_s % 60;
    
    std::ostringstream oss;
    if (h > 0) {
        oss << h << ":" << std::setfill('0') << std::setw(2) << m << ":" << std::setw(2) << s;
    } else {
        oss << m << ":" << std::setfill('0') << std::setw(2) << s;
    }
    return oss.str();
}

// Run multiple pools in parallel using a thread pool
template <typename T>
std::vector<PoolResult<T>> run_pools_parallel(
    const std::vector<std::pair<pools::PoolInit<T>, trading::Costs<T>>>& pool_configs,
    const std::vector<Event>& events,
    const RunConfig<T>& cfg,
    size_t n_threads = 0,
    bool verbose = true,
    const std::vector<Candle>* candles = nullptr
) {
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 1;
    }
    
    const size_t n_pools = pool_configs.size();
    std::vector<PoolResult<T>> results(n_pools);
    
    if (n_pools == 0) {
        return results;
    }
    
    // Log every ~1% of pools, minimum every pool if < 100, max every 1000
    const size_t log_interval = std::max(size_t(1), std::min(n_pools / 100, size_t(1000)));
    
    // Load cowswap trades if path specified
    std::vector<trading::CowswapTrade> cowswap_trades;
    const std::vector<trading::CowswapTrade>* cs_ptr = nullptr;
    if (!cfg.cowswap_path.empty()) {
        cowswap_trades = trading::load_cowswap_csv(cfg.cowswap_path);
        if (!cowswap_trades.empty()) {
            cs_ptr = &cowswap_trades;
            if (verbose) {
                std::lock_guard<std::mutex> lock(io_mu);
                std::cout << "loaded " << cowswap_trades.size() 
                          << " cowswap trades from " << cfg.cowswap_path << "\n" << std::flush;
            }
        }
    }
    
    // Track wall-clock time for ETA
    auto t_total_start = std::chrono::high_resolution_clock::now();
    
    // For single pool or single thread, run sequentially
    if (n_pools == 1 || n_threads == 1) {
        for (size_t i = 0; i < n_pools; ++i) {
            const auto& [pool_init, costs] = pool_configs[i];
            results[i] = run_single_pool(pool_init, costs, events, cfg, cs_ptr, candles);
            
            size_t done = i + 1;
            if (verbose && (done % log_interval == 0 || done == n_pools)) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - t_total_start).count();
                double avg_s = elapsed_s / done;
                double eta_s = avg_s * (n_pools - done);
                
                std::lock_guard<std::mutex> lock(io_mu);
                std::cout << "pool " << done << "/" << n_pools
                          << " (" << (100 * done / n_pools) << "%)"
                          << " | elapsed:" << format_duration(elapsed_s)
                          << " | eta:" << format_duration(eta_s)
                          << "\n" << std::flush;
            }
        }
        return results;
    }
    
    // Thread pool with work stealing via atomic index
    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> completed{0};
    std::atomic<size_t> last_logged{0};
    
    auto worker = [&]() {
        while (true) {
            const size_t i = next_idx.fetch_add(1);
            if (i >= n_pools) break;
            
            const auto& [pool_init, costs] = pool_configs[i];
            results[i] = run_single_pool(pool_init, costs, events, cfg, cs_ptr, candles);
            
            size_t done = completed.fetch_add(1) + 1;
            
            // Log at intervals or when complete
            if (verbose && (done == n_pools || done / log_interval > last_logged.load())) {
                last_logged.store(done / log_interval);
                
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - t_total_start).count();
                double avg_s = elapsed_s / done;
                double eta_s = avg_s * (n_pools - done);
                
                std::lock_guard<std::mutex> lock(io_mu);
                std::cout << "pool " << done << "/" << n_pools
                          << " (" << (100 * done / n_pools) << "%)"
                          << " | elapsed:" << format_duration(elapsed_s)
                          << " | eta:" << format_duration(eta_s)
                          << "\n" << std::flush;
            }
        }
    };
    
    // Launch worker threads
    const size_t actual_threads = std::min(n_threads, n_pools);
    std::vector<std::thread> threads;
    threads.reserve(actual_threads);
    
    for (size_t t = 0; t < actual_threads; ++t) {
        threads.emplace_back(worker);
    }
    
    // Wait for all to complete
    for (auto& th : threads) {
        th.join();
    }
    
    return results;
}

} // namespace harness
} // namespace arb
