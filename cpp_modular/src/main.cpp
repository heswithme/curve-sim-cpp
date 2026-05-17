// Modular arb_harness - Entry point with compile-time numeric type selection
//
// Build targets:
//   arb_harness    - double (default)
//   arb_harness_f  - float
//   arb_harness_ld - long double
//   arb_harness_pool_u256 - long double harness with uint256 pool

#include <iostream>
#include <chrono>
#include <vector>

#include "harness/cli.hpp"
#include "harness/runner.hpp"
#include "harness/output.hpp"
#include "harness/detailed_output.hpp"
#include "events/loader.hpp"
#include "pools/config.hpp"
#include "trading/costs.hpp"

// Compile-time numeric type selection.
#if defined(ARB_MODE_F)
using RealT = float;
using PoolT = float;
static constexpr const char* TYPE_NAME = "float";
#elif defined(ARB_MODE_LD)
using RealT = long double;
using PoolT = long double;
static constexpr const char* TYPE_NAME = "long double";
#elif defined(ARB_POOL_MODE_U256)
using RealT = long double;
using PoolT = arb::pools::twocrypto_fx::uint256;
static constexpr const char* TYPE_NAME = "pool_u256";
#else
using RealT = double;
using PoolT = double;
static constexpr const char* TYPE_NAME = "double";
#endif

int main(int argc, char* argv[]) {
    // Parse CLI arguments
    auto args = arb::harness::parse_cli(argc, argv);
    
    if (!args.valid) {
        if (argc < 2) {
            std::cout << "arb_harness_mod: " << TYPE_NAME << "\n";
            arb::harness::print_usage(argv[0]);
            return 0;
        }
        std::cerr << "Error: " << args.error_msg << "\n";
        arb::harness::print_usage(argv[0]);
        return 1;
    }

    try {
        auto t_read0 = std::chrono::high_resolution_clock::now();
        
        // Load candles and generate events
        auto candles = arb::load_candles(
            args.candles_path,
            args.max_candles,
            args.candle_filter_pct / 100.0,
            args.start_ts);
        const size_t n_candles = candles.size();
        auto events = arb::gen_events(candles);

        // Candle vector is only required for detailed per-event logging.
        // Free it eagerly for non-detailed runs to reduce memory footprint.
        const std::vector<arb::Candle>* candles_ptr = nullptr;
        if (args.detailed_log) {
            candles_ptr = &candles;
        } else {
            std::vector<arb::Candle>().swap(candles);
        }

        if (!args.quiet) {
            std::cout << "loaded " << n_candles << " candles -> "
                      << events.size() << " events from " << args.candles_path << "\n" << std::flush;
        }
        
        auto t_read1 = std::chrono::high_resolution_clock::now();
        double candles_read_ms = std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();

        // Load pool configs from JSON (optionally subset by contiguous or range-file assignment).
        std::vector<std::pair<arb::pools::PoolInit<PoolT, RealT>, arb::trading::Costs<RealT>>> pool_configs;
        if (!args.pool_ranges_path.empty()) {
            auto ranges = arb::pools::load_pool_ranges_file(args.pool_ranges_path);
            pool_configs = arb::pools::load_pool_configs_for_ranges<PoolT, RealT>(
                args.pools_path,
                ranges
            );
        } else {
            pool_configs = arb::pools::load_pool_configs<PoolT, RealT>(
                args.pools_path, args.pool_start, args.pool_end);
        }
        if (pool_configs.empty()) {
            throw std::runtime_error("No pool configurations found in " + args.pools_path);
        }
        if (!args.quiet) {
            std::cout << "loaded " << pool_configs.size() << " pools";
            if (!args.pool_ranges_path.empty()) {
                std::cout << " from ranges " << args.pool_ranges_path;
            } else if (args.pool_start > 0 || args.pool_end < SIZE_MAX) {
                std::cout << " (range " << args.pool_start << "-" << (args.pool_start + pool_configs.size()) << ")";
            }
            std::cout << "\n" << std::flush;
        }
        
        // Build run configuration from CLI args
        arb::harness::RunConfig<RealT> run_cfg{};
        run_cfg.min_swap_frac = static_cast<RealT>(args.min_swap_frac);
        run_cfg.max_swap_frac = static_cast<RealT>(args.max_swap_frac);
        run_cfg.start_ts = args.start_ts;
        run_cfg.dustswap_freq_s = args.dustswap_freq_s;
        run_cfg.user_swap_freq_s = args.user_swap_freq_s;
        run_cfg.user_swap_size_frac = static_cast<RealT>(args.user_swap_size_frac);
        run_cfg.user_swap_thresh = static_cast<RealT>(args.user_swap_thresh);
        run_cfg.enable_slippage_probes = !args.disable_slippage_probes;
        
        // Wire save_actions flag
        run_cfg.save_actions = args.save_actions;
        
        // Wire detailed_log flag
        run_cfg.detailed_log = args.detailed_log;
        run_cfg.detailed_interval = args.detailed_interval;
        
        // Wire cowswap trades path and fee
        run_cfg.cowswap_path = args.cowswap_path;
        run_cfg.cowswap_fee_bps = static_cast<RealT>(args.cowswap_fee_bps);
        
        auto t_exec0 = std::chrono::high_resolution_clock::now();
        
        // Run all pools in parallel
        auto results = arb::harness::run_pools_parallel(
            pool_configs, events, run_cfg,
            args.n_threads,
            !args.quiet,
            candles_ptr
        );
        
        auto t_exec1 = std::chrono::high_resolution_clock::now();
        double exec_ms = std::chrono::duration<double, std::milli>(t_exec1 - t_exec0).count();
        
        // Write output. A .json path preserves the legacy row JSON; otherwise
        // the path is treated as an arb_npz_v1 run directory.
        if (!args.out_path.empty()) {
            const bool legacy_json =
                args.out_path.size() >= 5 &&
                args.out_path.substr(args.out_path.size() - 5) == ".json";
            bool ok = legacy_json
                ? arb::harness::write_results_json(
                    args.out_path,
                    results,
                    n_candles,
                    events.size(),
                    args.candles_path,
                    args.pools_path,
                    TYPE_NAME,
                    args.n_threads,
                    run_cfg,
                    args.max_candles,
                    args.candle_filter_pct,
                    args.pool_start,
                    args.pool_end,
                    args.quiet,
                    candles_read_ms,
                    exec_ms
                )
                : arb::harness::write_results_npz_dir(
                    args.out_path,
                    results,
                    n_candles,
                    events.size(),
                    args.candles_path,
                    args.pools_path,
                    TYPE_NAME,
                    args.n_threads,
                    run_cfg,
                    args.max_candles,
                    args.candle_filter_pct,
                    args.pool_start,
                    args.pool_end,
                    args.quiet,
                    candles_read_ms,
                    exec_ms
                );
            if (!ok) {
                std::cerr << "Warning: Failed to write output to " << args.out_path << "\n";
            }
        }
        
        // Write detailed log if requested (uses first pool's detailed entries)
        // Place detailed-output next to the output file
        if (args.detailed_log && !results.empty()) {
            // Compute detailed output path: same directory as output, fixed name
            std::string detailed_log_path;
            const std::string detailed_log_name =
                args.detailed_npz ? "detailed-output.npz" : "detailed-output.json";
            {
                auto pos = args.out_path.find_last_of("/\\");
                if (pos != std::string::npos) {
                    detailed_log_path = args.out_path.substr(0, pos + 1) + detailed_log_name;
                } else {
                    detailed_log_path = detailed_log_name;
                }
            }
            
            // Find first successful result with detailed entries
            for (const auto& res : results) {
                if (res.success && !res.detailed_entries.empty()) {
                    bool ok = args.detailed_npz
                        ? arb::harness::write_detailed_npz(detailed_log_path, res.detailed_entries)
                        : arb::harness::write_detailed_log(detailed_log_path, res.detailed_entries);
                    if (!ok) {
                        std::cerr << "Warning: Failed to write detailed log to "
                                  << detailed_log_path << "\n";
                    } else {
                        std::cout << "Wrote detailed "
                                  << (args.detailed_npz ? "NPZ log" : "log")
                                  << " (" << res.detailed_entries.size()
                                  << " entries) to " << detailed_log_path << "\n";
                    }
                    break;  // Only write first pool's detailed log
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
