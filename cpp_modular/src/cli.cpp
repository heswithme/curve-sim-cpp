// CLI argument parsing implementation

#include "harness/cli.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace arb {
namespace harness {

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name
              << " <pools.json> <candles.json> <output.json>\n"
              << "       [--n-candles N] [--save-actions]\n"
              << "       [--min-swap F] [--max-swap F]\n"
              << "       [--threads N | -n N] [--candle-filter PCT]\n"
              << "       [--dustswapfreq S]\n"
              << "       [--pool-start N] [--pool-end N]\n"
              << "       [--userswapfreq S] [--userswapsize F] [--userswapthresh F]\n"
              << "       [--detailed-log] [--detailed-interval N] [--disable-slippage-probes]\n"
              << "       [--cowswap-trades PATH] [--cowswap-fee-bps BPS]\n";
}

CliArgs parse_cli(int argc, char* argv[]) {
    CliArgs args{};

    if (argc < 4) {
        args.valid = false;
        args.error_msg = "Not enough arguments (need pools.json, candles.json, output.json)";
        return args;
    }

    args.pools_path = argv[1];
    args.candles_path = argv[2];
    args.out_path = argv[3];

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        try {
            if (arg == "--n-candles" && i + 1 < argc) {
                args.max_candles = static_cast<size_t>(std::stoll(argv[++i]));
            } else if (arg == "--save-actions") {
                args.save_actions = true;
            } else if (arg == "--min-swap" && i + 1 < argc) {
                args.min_swap_frac = std::stod(argv[++i]);
            } else if (arg == "--max-swap" && i + 1 < argc) {
                args.max_swap_frac = std::stod(argv[++i]);
            } else if ((arg == "--threads" || arg == "-n") && i + 1 < argc) {
                args.n_threads = static_cast<size_t>(std::stoll(argv[++i]));
            } else if (arg == "--candle-filter" && i + 1 < argc) {
                args.candle_filter_pct = std::stod(argv[++i]);
            } else if (arg == "--dustswapfreq" && i + 1 < argc) {
                args.dustswap_freq_s = static_cast<uint64_t>(std::stoll(argv[++i]));
            } else if (arg == "--pool-start" && i + 1 < argc) {
                args.pool_start = static_cast<size_t>(std::stoll(argv[++i]));
            } else if (arg == "--pool-end" && i + 1 < argc) {
                args.pool_end = static_cast<size_t>(std::stoll(argv[++i]));
            } else if (arg == "--userswapfreq" && i + 1 < argc) {
                args.user_swap_freq_s = static_cast<uint64_t>(std::stoll(argv[++i]));
            } else if (arg == "--userswapsize" && i + 1 < argc) {
                args.user_swap_size_frac = std::stod(argv[++i]);
            } else if (arg == "--userswapthresh" && i + 1 < argc) {
                args.user_swap_thresh = std::stod(argv[++i]);
            } else if (arg == "--detailed-log") {
                args.detailed_log = true;
            } else if (arg == "--detailed-interval" && i + 1 < argc) {
                args.detailed_interval = static_cast<size_t>(std::stoll(argv[++i]));
                if (args.detailed_interval == 0) args.detailed_interval = 1;
            } else if (arg == "--disable-slippage-probes") {
                args.disable_slippage_probes = true;
            } else if (arg == "--cowswap-trades" && i + 1 < argc) {
                args.cowswap_path = argv[++i];
            } else if (arg == "--cowswap-fee-bps" && i + 1 < argc) {
                args.cowswap_fee_bps = std::stod(argv[++i]);
            }
            // Unknown flags are silently ignored (matches original behavior)
        } catch (...) {
            // Ignore parse errors for individual flags (matches original behavior)
        }
    }

    // Clamp negative values
    if (args.user_swap_size_frac < 0) args.user_swap_size_frac = 0;
    if (args.user_swap_thresh < 0) args.user_swap_thresh = 0;
    if (args.n_threads == 0) args.n_threads = 1;
    
    // Validate parameters
    if (args.min_swap_frac < 0) {
        args.valid = false;
        args.error_msg = "--min-swap must be >= 0";
        return args;
    }
    if (args.max_swap_frac < 0) {
        args.valid = false;
        args.error_msg = "--max-swap must be >= 0";
        return args;
    }
    if (args.max_swap_frac < args.min_swap_frac) {
        args.valid = false;
        args.error_msg = "--max-swap must be >= --min-swap";
        return args;
    }
    if (args.cowswap_fee_bps < 0) {
        args.valid = false;
        args.error_msg = "--cowswap-fee-bps must be >= 0";
        return args;
    }
    if (args.candle_filter_pct < 0) {
        args.valid = false;
        args.error_msg = "--candle-filter must be >= 0";
        return args;
    }

    args.valid = true;
    return args;
}

} // namespace harness
} // namespace arb
