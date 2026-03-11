// Modular arb_harness - runtime-selectable pool backend

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>

#include "events/loader.hpp"
#include "harness/cli.hpp"
#include "harness/detailed_output.hpp"
#include "harness/output.hpp"
#include "harness/runtime_backend.hpp"
#include "harness/runtime_runner.hpp"
#include "pools/config.hpp"
#include "pools/twocrypto_fx/helpers.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"
#include "trading/arbitrageur.hpp"

template <typename T>
void print_value(const char* name, const T& val) {
    std::cout << "  " << name << " = " << std::setprecision(12) << val << "\n";
}

template <typename T>
void test_pool(T cex_price = T(-1)) {
    using Pool = arb::pools::twocrypto_fx::TwoCryptoPool<T>;
    using Traits = arb::pools::twocrypto_fx::PoolTraits<T>;

    const std::array<T, 2> precisions = {Traits::ONE(), Traits::ONE()};
    const T A = T(10000.0);
    const T gamma = T(1e-5);
    const T mid_fee = T(0.0001);
    const T out_fee = T(0.0006);
    const T fee_gamma = T(0.00023);
    const T allowed_extra_profit = T(1e-8);
    const T adjustment_step = T(0.0001);
    const T ma_time = T(600.0);
    const T initial_price = T(1.08);

    Pool pool(
        precisions,
        A,
        gamma,
        mid_fee,
        out_fee,
        fee_gamma,
        allowed_extra_profit,
        adjustment_step,
        ma_time,
        initial_price
    );

    pool.set_block_timestamp(1700000000);

    std::cout << "Pool created with initial_price:\n";
    print_value("cached_price_scale", pool.cached_price_scale);
    print_value("cached_price_oracle", pool.cached_price_oracle);

    const T amount0 = T(10000.0);
    const T amount1 = T(10000.0 / 1.08);
    const std::array<T, 2> amounts = {amount0, amount1};
    const T lp_tokens = pool.add_liquidity(amounts, Traits::ZERO());

    std::cout << "\nAfter add_liquidity:\n";
    print_value("LP tokens minted", lp_tokens);
    print_value("totalSupply", pool.totalSupply);
    print_value("D", pool.D);
    print_value("balances[0]", pool.balances[0]);
    print_value("balances[1]", pool.balances[1]);
    print_value("virtual_price", pool.get_virtual_price());

    const T dx = T(100.0);
    const auto sim = arb::pools::twocrypto_fx::simulate_exchange_once(pool, 0, 1, dx);
    std::cout << "\nSimulated exchange (0 -> 1):\n";
    print_value("dx", dx);
    print_value("dy_after_fee", sim.first);
    print_value("fee_tokens", sim.second);

    if (cex_price > T(0)) {
        arb::trading::Costs<T> costs{};
        const auto dec = arb::trading::decide_trade(
            pool,
            cex_price,
            costs,
            std::numeric_limits<T>::infinity(),
            T(0.001),
            T(0.1)
        );
        std::cout << "\nArb decision:\n";
        std::cout << "  do_trade = " << dec.do_trade << "\n";
        print_value("  dx", dec.dx);
        print_value("  profit_coin0", dec.profit);
        print_value("  fee_tokens", dec.fee_tokens);
    }

    if (pool.D > Traits::ZERO()) {
        std::cout << "\nPool test: PASSED (D > 0)\n";
    } else {
        std::cout << "\nPool test: FAILED (D <= 0)\n";
    }
}

int main(int argc, char* argv[]) {
    auto args = arb::harness::parse_cli(argc, argv);

    if (!args.valid) {
        if (argc < 2) {
            std::cout << "arb_harness_mod: runtime\n";
            std::cout << "\n--- Pool Test ---\n";
            test_pool<double>();
            arb::harness::print_usage(argv[0]);
            return 0;
        }
        std::cerr << "Error: " << args.error_msg << "\n";
        arb::harness::print_usage(argv[0]);
        return 1;
    }

    arb::harness::PoolBackend backend = arb::harness::PoolBackend::Double;
    if (!arb::harness::parse_pool_backend(args.pool_backend, backend)) {
        std::cerr << "Error: invalid --pool-backend: " << args.pool_backend << "\n";
        arb::harness::print_usage(argv[0]);
        return 1;
    }

    try {
        auto t_read0 = std::chrono::high_resolution_clock::now();

        auto candles = arb::load_candles(
            args.candles_path,
            args.max_candles,
            args.candle_filter_pct / 100.0
        );
        const size_t n_candles = candles.size();
        auto events = arb::gen_events(candles);

        const std::vector<arb::Candle>* candles_ptr = nullptr;
        if (args.detailed_log) {
            candles_ptr = &candles;
        } else {
            std::vector<arb::Candle>().swap(candles);
        }

        std::cout << "loaded " << n_candles << " candles -> "
                  << events.size() << " events from " << args.candles_path
                  << " using pool_backend=" << arb::harness::pool_backend_name(backend)
                  << "\n" << std::flush;

        auto t_read1 = std::chrono::high_resolution_clock::now();
        const double candles_read_ms =
            std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();

        auto pool_configs = arb::pools::load_pool_configs<double>(
            args.pools_path,
            args.pool_start,
            args.pool_end
        );
        if (pool_configs.empty()) {
            throw std::runtime_error("No pool configurations found in " + args.pools_path);
        }
        std::cout << "loaded " << pool_configs.size() << " pools";
        if (args.pool_start > 0 || args.pool_end < SIZE_MAX) {
            std::cout << " (range " << args.pool_start << "-"
                      << (args.pool_start + pool_configs.size()) << ")";
        }
        std::cout << "\n" << std::flush;

        arb::harness::RuntimeRunConfig run_cfg{};
        run_cfg.min_swap_frac = args.min_swap_frac;
        run_cfg.max_swap_frac = args.max_swap_frac;
        run_cfg.dustswap_freq_s = args.dustswap_freq_s;
        run_cfg.user_swap_freq_s = args.user_swap_freq_s;
        run_cfg.user_swap_size_frac = args.user_swap_size_frac;
        run_cfg.user_swap_thresh = args.user_swap_thresh;
        run_cfg.enable_slippage_probes = !args.disable_slippage_probes;
        run_cfg.save_actions = args.save_actions;
        run_cfg.detailed_log = args.detailed_log;
        run_cfg.detailed_interval = args.detailed_interval;
        run_cfg.cowswap_path = args.cowswap_path;
        run_cfg.cowswap_fee_bps = args.cowswap_fee_bps;

        auto t_exec0 = std::chrono::high_resolution_clock::now();
        const auto results = arb::harness::run_pools_parallel_runtime(
            pool_configs,
            events,
            run_cfg,
            backend,
            args.n_threads,
            true,
            candles_ptr
        );
        auto t_exec1 = std::chrono::high_resolution_clock::now();
        const double exec_ms =
            std::chrono::duration<double, std::milli>(t_exec1 - t_exec0).count();

        if (!args.out_path.empty()) {
            const bool ok = arb::harness::write_results_json(
                args.out_path,
                results,
                events.size(),
                args.candles_path,
                args.n_threads,
                candles_read_ms,
                exec_ms
            );
            if (!ok) {
                std::cerr << "Warning: Failed to write output to " << args.out_path << "\n";
            }
        }

        if (args.detailed_log && !results.empty()) {
            std::string detailed_log_path;
            const auto pos = args.out_path.find_last_of("/\\");
            if (pos != std::string::npos) {
                detailed_log_path = args.out_path.substr(0, pos + 1) + "detailed-output.json";
            } else {
                detailed_log_path = "detailed-output.json";
            }

            for (const auto& res : results) {
                if (res.success && !res.detailed_entries.empty()) {
                    const bool ok = arb::harness::write_detailed_log(
                        detailed_log_path,
                        res.detailed_entries
                    );
                    if (!ok) {
                        std::cerr << "Warning: Failed to write detailed log to "
                                  << detailed_log_path << "\n";
                    } else {
                        std::cout << "Wrote detailed log (" << res.detailed_entries.size()
                                  << " entries) to " << detailed_log_path << "\n";
                    }
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
