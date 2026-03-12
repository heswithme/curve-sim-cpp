#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/json.hpp>

#include "core/json_utils.hpp"
#include "events/loader.hpp"
#include "harness/output.hpp"
#include "harness/runner.hpp"
#include "pools/config.hpp"
#include "trading/costs.hpp"

namespace json = boost::json;

namespace {

using RealT = double;

struct CliArgs {
    std::string template_path;
    std::string candles_path;
    std::size_t pool_index{0};
    std::size_t max_candles{0};
    double candle_filter_pct{99.0};

    double min_swap{1e-6};
    double max_swap{1.0};
    uint64_t dustswap_freq_s{3600};
    uint64_t user_swap_freq_s{0};
    double user_swap_size_frac{0.01};
    double user_swap_thresh{0.05};
    bool disable_slippage_probes{false};

    std::string request_json;

    bool valid{false};
    std::string error;
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <template_pools.json> <candles.json> [options]\n\n"
        << "Options:\n"
        << "  --pool-index N             Pool entry index from template file (default: 0)\n"
        << "  --n-candles N              Limit candles loaded (default: all)\n"
        << "  --candle-filter PCT        Candle squeeze percent (default: 99)\n"
        << "  --min-swap X               Min swap fraction for arb search (default: 1e-6)\n"
        << "  --max-swap X               Max swap fraction for arb search (default: 1.0)\n"
        << "  --dustswapfreq S           Dust swap interval seconds (default: 3600)\n"
        << "  --userswapfreq S           User swap interval seconds (default: 0=off)\n"
        << "  --userswapsize F           User swap size fraction (default: 0.01)\n"
        << "  --userswapthresh F         User swap threshold (default: 0.05)\n"
        << "  --disable-slippage-probes  Disable slippage probes\n"
        << "  --request JSON             Run one request and exit\n"
        << "  -h, --help                 Show this help\n\n"
        << "Stdin protocol (JSONL in persistent mode):\n"
        << "  request fields:\n"
        << "    id (optional, echoed), mid_fee or mid_fee_bps, out_fee or out_fee_bps, fee_gamma\n"
        << "  response fields:\n"
        << "    ok, vp, apy, apy_net, avg_rel_price_diff, max_rel_price_diff, elapsed_ms\n";
}

bool parse_size_t(const std::string& s, std::size_t& out) {
    try {
        const auto v = std::stoull(s);
        out = static_cast<std::size_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_u64(const std::string& s, uint64_t& out) {
    try {
        out = static_cast<uint64_t>(std::stoull(s));
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& s, double& out) {
    try {
        out = std::stod(s);
        return std::isfinite(out);
    } catch (...) {
        return false;
    }
}

std::string format_double_plain(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss << std::setprecision(15) << v;
    std::string s = oss.str();

    const auto dot_pos = s.find('.');
    if (dot_pos != std::string::npos) {
        while (!s.empty() && s.back() == '0') {
            s.pop_back();
        }
        if (!s.empty() && s.back() == '.') {
            s.pop_back();
        }
    }

    if (s == "-0") {
        return "0";
    }
    if (s.empty()) {
        return "0";
    }
    return s;
}

void write_json_plain(std::ostream& os, const json::value& v) {
    switch (v.kind()) {
        case json::kind::null:
            os << "null";
            return;
        case json::kind::bool_:
            os << (v.as_bool() ? "true" : "false");
            return;
        case json::kind::int64:
            os << v.as_int64();
            return;
        case json::kind::uint64:
            os << v.as_uint64();
            return;
        case json::kind::double_:
            os << format_double_plain(v.as_double());
            return;
        case json::kind::string:
            os << json::serialize(v);
            return;
        case json::kind::array: {
            const auto& arr = v.as_array();
            os << '[';
            for (std::size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) {
                    os << ',';
                }
                write_json_plain(os, arr[i]);
            }
            os << ']';
            return;
        }
        case json::kind::object: {
            const auto& obj = v.as_object();
            os << '{';
            bool first = true;
            for (const auto& kv : obj) {
                if (!first) {
                    os << ',';
                }
                first = false;
                os << json::serialize(json::value(kv.key())) << ':';
                write_json_plain(os, kv.value());
            }
            os << '}';
            return;
        }
    }
}

void write_json_line(const json::object& obj) {
    write_json_plain(std::cout, obj);
    std::cout << '\n' << std::flush;
}

CliArgs parse_cli(int argc, char* argv[]) {
    CliArgs args;

    if (argc >= 2) {
        const std::string a1 = argv[1];
        if (a1 == "-h" || a1 == "--help") {
            args.valid = false;
            return args;
        }
    }

    if (argc < 3) {
        args.error = "Not enough arguments";
        return args;
    }

    args.template_path = argv[1];
    args.candles_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        const std::string opt = argv[i];
        auto next_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                args.error = std::string("Missing value for ") + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (opt == "--pool-index") {
            const char* v = next_value("--pool-index");
            if (!v) return args;
            if (!parse_size_t(v, args.pool_index)) {
                args.error = "Invalid --pool-index";
                return args;
            }
        } else if (opt == "--n-candles") {
            const char* v = next_value("--n-candles");
            if (!v) return args;
            if (!parse_size_t(v, args.max_candles)) {
                args.error = "Invalid --n-candles";
                return args;
            }
        } else if (opt == "--candle-filter") {
            const char* v = next_value("--candle-filter");
            if (!v) return args;
            if (!parse_double(v, args.candle_filter_pct)) {
                args.error = "Invalid --candle-filter";
                return args;
            }
        } else if (opt == "--min-swap") {
            const char* v = next_value("--min-swap");
            if (!v) return args;
            if (!parse_double(v, args.min_swap)) {
                args.error = "Invalid --min-swap";
                return args;
            }
        } else if (opt == "--max-swap") {
            const char* v = next_value("--max-swap");
            if (!v) return args;
            if (!parse_double(v, args.max_swap)) {
                args.error = "Invalid --max-swap";
                return args;
            }
        } else if (opt == "--dustswapfreq") {
            const char* v = next_value("--dustswapfreq");
            if (!v) return args;
            if (!parse_u64(v, args.dustswap_freq_s)) {
                args.error = "Invalid --dustswapfreq";
                return args;
            }
        } else if (opt == "--userswapfreq") {
            const char* v = next_value("--userswapfreq");
            if (!v) return args;
            if (!parse_u64(v, args.user_swap_freq_s)) {
                args.error = "Invalid --userswapfreq";
                return args;
            }
        } else if (opt == "--userswapsize") {
            const char* v = next_value("--userswapsize");
            if (!v) return args;
            if (!parse_double(v, args.user_swap_size_frac)) {
                args.error = "Invalid --userswapsize";
                return args;
            }
        } else if (opt == "--userswapthresh") {
            const char* v = next_value("--userswapthresh");
            if (!v) return args;
            if (!parse_double(v, args.user_swap_thresh)) {
                args.error = "Invalid --userswapthresh";
                return args;
            }
        } else if (opt == "--disable-slippage-probes") {
            args.disable_slippage_probes = true;
        } else if (opt == "--request") {
            const char* v = next_value("--request");
            if (!v) return args;
            args.request_json = v;
        } else if (opt == "-h" || opt == "--help") {
            args.valid = false;
            return args;
        } else {
            args.error = "Unknown option: " + opt;
            return args;
        }
    }

    if (args.min_swap < 0.0) {
        args.error = "--min-swap must be >= 0";
        return args;
    }
    if (args.max_swap < 0.0) {
        args.error = "--max-swap must be >= 0";
        return args;
    }
    if (args.max_swap < args.min_swap) {
        args.error = "--max-swap must be >= --min-swap";
        return args;
    }
    if (args.candle_filter_pct < 0.0) {
        args.error = "--candle-filter must be >= 0";
        return args;
    }

    if (args.user_swap_size_frac < 0.0) args.user_swap_size_frac = 0.0;
    if (args.user_swap_thresh < 0.0) args.user_swap_thresh = 0.0;

    args.valid = true;
    return args;
}

double get_double_opt(const json::object& obj, const char* key, double fallback) {
    const auto it = obj.find(key);
    if (it == obj.end()) return fallback;
    const auto& v = it->value();
    if (v.is_double()) return v.as_double();
    if (v.is_int64()) return static_cast<double>(v.as_int64());
    if (v.is_uint64()) return static_cast<double>(v.as_uint64());
    if (v.is_string()) {
        try {
            return std::stod(std::string(v.as_string().c_str()));
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

json::object make_error_response(const json::object& req, const std::string& err) {
    json::object out;
    if (auto* id = req.if_contains("id")) {
        out["id"] = *id;
    }
    out["ok"] = false;
    out["error"] = err;
    return out;
}

json::object evaluate_request(
    const json::object& req,
    const arb::pools::PoolInit<RealT>& base_pool,
    const arb::trading::Costs<RealT>& base_costs,
    const std::vector<arb::Event>& events,
    const arb::harness::RunConfig<RealT>& run_cfg
) {
    auto pool = base_pool;
    auto costs = base_costs;

    const double mid_fee = [&]() {
        if (req.if_contains("mid_fee")) {
            return get_double_opt(req, "mid_fee", static_cast<double>(pool.mid_fee));
        }
        if (req.if_contains("mid_fee_bps")) {
            return get_double_opt(req, "mid_fee_bps", static_cast<double>(pool.mid_fee) * 10000.0) / 10000.0;
        }
        return static_cast<double>(pool.mid_fee);
    }();

    const double out_fee = [&]() {
        if (req.if_contains("out_fee")) {
            return get_double_opt(req, "out_fee", static_cast<double>(pool.out_fee));
        }
        if (req.if_contains("out_fee_bps")) {
            return get_double_opt(req, "out_fee_bps", static_cast<double>(pool.out_fee) * 10000.0) / 10000.0;
        }
        return static_cast<double>(pool.out_fee);
    }();

    const double fee_gamma = req.if_contains("fee_gamma")
        ? get_double_opt(req, "fee_gamma", static_cast<double>(pool.fee_gamma))
        : static_cast<double>(pool.fee_gamma);

    if (!std::isfinite(mid_fee) || !std::isfinite(out_fee) || !std::isfinite(fee_gamma)) {
        return make_error_response(req, "Non-finite parameter value");
    }
    if (mid_fee < 0.0 || out_fee < 0.0 || fee_gamma < 0.0) {
        return make_error_response(req, "Fees and fee_gamma must be >= 0");
    }
    if (mid_fee > 1.0 || out_fee > 1.0 || fee_gamma > 1.0) {
        return make_error_response(req, "Fees and fee_gamma must be <= 1");
    }
    if (out_fee < mid_fee) {
        return make_error_response(req, "out_fee must be >= mid_fee");
    }

    pool.mid_fee = static_cast<RealT>(mid_fee);
    pool.out_fee = static_cast<RealT>(out_fee);
    pool.fee_gamma = static_cast<RealT>(fee_gamma);

    if (req.if_contains("A")) {
        pool.A = static_cast<RealT>(get_double_opt(req, "A", static_cast<double>(pool.A)));
    }
    if (req.if_contains("gamma")) {
        pool.gamma = static_cast<RealT>(get_double_opt(req, "gamma", static_cast<double>(pool.gamma)));
    }
    if (req.if_contains("lp_profit_fraction")) {
        const double fraction = get_double_opt(
            req,
            "lp_profit_fraction",
            static_cast<double>(pool.lp_profit_fraction)
        );
        pool.lp_profit_fraction = static_cast<RealT>(std::clamp(fraction, 0.0, 1.0));
    }
    if (req.if_contains("allowed_extra_profit")) {
        pool.allowed_extra_profit = static_cast<RealT>(
            get_double_opt(req, "allowed_extra_profit", static_cast<double>(pool.allowed_extra_profit))
        );
    }
    if (req.if_contains("adjustment_step")) {
        pool.adjustment_step = static_cast<RealT>(
            get_double_opt(req, "adjustment_step", static_cast<double>(pool.adjustment_step))
        );
    }
    if (req.if_contains("ma_time")) {
        pool.ma_time = static_cast<RealT>(get_double_opt(req, "ma_time", static_cast<double>(pool.ma_time)));
    }
    if (req.if_contains("donation_apy")) {
        pool.donation_apy = static_cast<RealT>(
            get_double_opt(req, "donation_apy", static_cast<double>(pool.donation_apy))
        );
    }
    if (req.if_contains("donation_frequency")) {
        pool.donation_frequency = static_cast<RealT>(
            get_double_opt(req, "donation_frequency", static_cast<double>(pool.donation_frequency))
        );
    }
    if (req.if_contains("donation_coins_ratio")) {
        const double ratio = get_double_opt(req, "donation_coins_ratio", static_cast<double>(pool.donation_coins_ratio));
        pool.donation_coins_ratio = static_cast<RealT>(std::clamp(ratio, 0.0, 1.0));
    }

    if (req.if_contains("arb_fee_bps")) {
        costs.arb_fee_bps = static_cast<RealT>(get_double_opt(req, "arb_fee_bps", static_cast<double>(costs.arb_fee_bps)));
    }
    if (req.if_contains("gas_coin0")) {
        costs.gas_coin0 = static_cast<RealT>(get_double_opt(req, "gas_coin0", static_cast<double>(costs.gas_coin0)));
    }

    const auto run = arb::harness::run_single_pool(pool, costs, events, run_cfg, nullptr, nullptr);
    if (!run.success) {
        return make_error_response(req, run.error_msg.empty() ? "run_single_pool failed" : run.error_msg);
    }

    const auto summary = arb::harness::metrics_to_summary(run, events.size());

    json::object out;
    if (auto* id = req.if_contains("id")) {
        out["id"] = *id;
    }
    out["ok"] = true;
    out["mid_fee"] = static_cast<double>(pool.mid_fee);
    out["out_fee"] = static_cast<double>(pool.out_fee);
    out["fee_gamma"] = static_cast<double>(pool.fee_gamma);
    out["lp_profit_fraction"] = static_cast<double>(pool.lp_profit_fraction);
    out["mid_fee_bps"] = static_cast<double>(pool.mid_fee) * 10000.0;
    out["out_fee_bps"] = static_cast<double>(pool.out_fee) * 10000.0;

    out["vp"] = get_double_opt(summary, "vp", -1.0);
    out["apy"] = get_double_opt(summary, "apy", -1.0);
    out["apy_net"] = get_double_opt(summary, "apy_net", -1.0);
    out["avg_rel_price_diff"] = get_double_opt(summary, "avg_rel_price_diff", -1.0);
    out["max_rel_price_diff"] = get_double_opt(summary, "max_rel_price_diff", -1.0);

    out["trades"] = get_double_opt(summary, "trades", 0.0);
    out["n_rebalances"] = get_double_opt(summary, "n_rebalances", 0.0);
    out["elapsed_ms"] = run.elapsed_ms;
    return out;
}

}  // namespace

int main(int argc, char* argv[]) {
    const CliArgs args = parse_cli(argc, argv);
    if (!args.valid) {
        print_usage(argv[0]);
        if (!args.error.empty()) {
            std::cerr << "\nError: " << args.error << "\n";
            return 1;
        }
        return 0;
    }

    try {
        auto candles = arb::load_candles(
            args.candles_path,
            args.max_candles,
            args.candle_filter_pct / 100.0
        );
        auto events = arb::gen_events(candles);
        if (events.empty()) {
            throw std::runtime_error("No events generated from candles");
        }

        auto templates = arb::pools::load_pool_configs<RealT>(args.template_path);
        if (templates.empty()) {
            throw std::runtime_error("No pool templates found in " + args.template_path);
        }
        if (args.pool_index >= templates.size()) {
            throw std::runtime_error(
                "pool-index out of range (" + std::to_string(args.pool_index) +
                " >= " + std::to_string(templates.size()) + ")"
            );
        }

        const auto& base_pool = templates[args.pool_index].first;
        const auto& base_costs = templates[args.pool_index].second;

        arb::harness::RunConfig<RealT> run_cfg{};
        run_cfg.min_swap_frac = static_cast<RealT>(args.min_swap);
        run_cfg.max_swap_frac = static_cast<RealT>(args.max_swap);
        run_cfg.dustswap_freq_s = args.dustswap_freq_s;
        run_cfg.user_swap_freq_s = args.user_swap_freq_s;
        run_cfg.user_swap_size_frac = static_cast<RealT>(args.user_swap_size_frac);
        run_cfg.user_swap_thresh = static_cast<RealT>(args.user_swap_thresh);
        run_cfg.enable_slippage_probes = !args.disable_slippage_probes;
        run_cfg.save_actions = false;
        run_cfg.detailed_log = false;

        auto handle_one_request = [&](const std::string& raw) {
            json::object req_obj;
            try {
                json::value req = json::parse(raw);
                if (!req.is_object()) {
                    json::object err;
                    err["ok"] = false;
                    err["error"] = "Request must be a JSON object";
                    write_json_line(err);
                    return;
                }
                req_obj = req.as_object();
            } catch (const std::exception& e) {
                json::object err;
                err["ok"] = false;
                err["error"] = std::string("JSON parse error: ") + e.what();
                write_json_line(err);
                return;
            }

            json::object response = evaluate_request(req_obj, base_pool, base_costs, events, run_cfg);
            write_json_line(response);
        };

        if (!args.request_json.empty()) {
            handle_one_request(args.request_json);
            return 0;
        }

        std::cerr
            << "ready: loaded " << candles.size() << " candles, "
            << events.size() << " events, using template index "
            << args.pool_index << '\n';

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            if (line == "exit" || line == "quit") {
                break;
            }
            handle_one_request(line);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
