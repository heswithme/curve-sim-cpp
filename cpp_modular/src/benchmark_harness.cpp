#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/json.hpp>

#include "pools/twocrypto_fx/twocrypto.hpp"

namespace json = boost::json;

namespace {

using arb::pools::twocrypto_fx::MathOps;
using arb::pools::twocrypto_fx::PoolTraits;
using arb::pools::twocrypto_fx::TwoCryptoPool;
using arb::pools::twocrypto_fx::uint256;

constexpr long double WAD = 1e18L;
constexpr long double FEE_SCALE = 1e10L;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

uint64_t env_u64(const char* key, uint64_t default_value) {
    if (const char* v = std::getenv(key)) {
        try {
            return static_cast<uint64_t>(std::stoull(v));
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

bool env_flag(const char* key) {
    if (const char* v = std::getenv(key)) {
        return std::string(v) == "1";
    }
    return false;
}

// JSON helpers
std::string get_str(const json::object& obj, const char* key) {
    auto it = obj.find(key);
    if (it == obj.end()) {
        throw std::runtime_error(std::string("missing key: ") + key);
    }
    if (!it->value().is_string()) {
        throw std::runtime_error(std::string("expected string for key: ") + key);
    }
    return std::string(it->value().as_string().c_str());
}

uint64_t get_u64_opt(const json::object& obj, const char* key, uint64_t default_value) {
    auto it = obj.find(key);
    if (it == obj.end()) return default_value;
    const auto& v = it->value();
    if (v.is_uint64()) return v.as_uint64();
    if (v.is_int64()) return static_cast<uint64_t>(v.as_int64());
    if (v.is_string()) {
        try {
            return static_cast<uint64_t>(std::stoull(std::string(v.as_string().c_str())));
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

template <typename T>
T parse_raw(const std::string& s) {
    if constexpr (std::is_same_v<T, uint256>) {
        return uint256(s);
    } else {
        return static_cast<T>(std::strtold(s.c_str(), nullptr));
    }
}

template <typename T>
T parse_wad(const std::string& s) {
    if constexpr (std::is_same_v<T, uint256>) {
        return uint256(s);
    } else {
        return static_cast<T>(std::strtold(s.c_str(), nullptr) / WAD);
    }
}

template <typename T>
T parse_fee(const std::string& s) {
    if constexpr (std::is_same_v<T, uint256>) {
        return uint256(s);
    } else {
        return static_cast<T>(std::strtold(s.c_str(), nullptr) / FEE_SCALE);
    }
}

template <typename T>
std::string to_int_string(const T& v) {
    if constexpr (std::is_same_v<T, uint256>) {
        return v.template convert_to<std::string>();
    } else {
        long double x = static_cast<long double>(v);
        if (!std::isfinite(x)) x = 0;
        if (x < 0) x = 0;
        const auto rounded = std::floor(x + 0.5L);
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(0);
        oss << rounded;
        return oss.str();
    }
}

template <typename T>
std::string to_wei_string(const T& v) {
    if constexpr (std::is_same_v<T, uint256>) {
        return v.template convert_to<std::string>();
    } else {
        long double x = static_cast<long double>(v);
        if (!std::isfinite(x)) x = 0;
        if (x < 0) x = 0;
        const auto scaled = std::floor(x * WAD + 0.5L);
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(0);
        oss << scaled;
        return oss.str();
    }
}

template <typename T>
json::object snapshot_pool(const TwoCryptoPool<T>& pool) {
    using Traits = PoolTraits<T>;

    const std::array<T, 2> xp = {
        pool.balances[0] * pool.precisions[0],
        (pool.balances[1] * pool.precisions[1] * pool.cached_price_scale) / Traits::PRECISION(),
    };

    json::object o;
    o["balances"] = json::array{to_wei_string(pool.balances[0]), to_wei_string(pool.balances[1])};
    o["xp"] = json::array{to_wei_string(xp[0]), to_wei_string(xp[1])};
    o["D"] = to_wei_string(pool.D);
    o["virtual_price"] = to_wei_string(pool.get_virtual_price());
    o["xcp_profit"] = to_wei_string(pool.xcp_profit);
    o["price_scale"] = to_wei_string(pool.cached_price_scale);
    o["price_oracle"] = to_wei_string(pool.cached_price_oracle);
    o["last_prices"] = to_wei_string(pool.last_prices);
    o["totalSupply"] = to_wei_string(pool.totalSupply);
    o["timestamp"] = pool.block_timestamp;

    o["donation_shares"] = to_wei_string(pool.donation_shares);
    o["donation_shares_unlocked"] = to_wei_string(pool.donation_unlocked());

    o["donation_protection_expiry_ts"] = to_int_string(pool.donation_protection_expiry_ts);
    o["last_donation_release_ts"] = to_int_string(pool.last_donation_release_ts);

    return o;
}

template <typename T>
TwoCryptoPool<T> make_pool_from_json(const json::object& p, const json::object& sequence) {
    const std::array<T, 2> precisions = {PoolTraits<T>::ONE(), PoolTraits<T>::ONE()};

    const T A = parse_raw<T>(get_str(p, "A"));
    const T gamma = parse_wad<T>(get_str(p, "gamma"));

    const T mid_fee = parse_fee<T>(get_str(p, "mid_fee"));
    const T out_fee = parse_fee<T>(get_str(p, "out_fee"));
    const T fee_gamma = parse_wad<T>(get_str(p, "fee_gamma"));

    const T allowed_extra_profit = parse_wad<T>(get_str(p, "allowed_extra_profit"));
    const T adjustment_step = parse_wad<T>(get_str(p, "adjustment_step"));

    const T ma_time = parse_raw<T>(get_str(p, "ma_time"));
    const T initial_price = parse_wad<T>(get_str(p, "initial_price"));

    TwoCryptoPool<T> pool(
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

    const uint64_t start_ts = get_u64_opt(sequence, "start_timestamp", 0);
    if (start_ts > 0) {
        pool.set_block_timestamp(start_ts);
    }

    auto it = p.find("initial_liquidity");
    if (it == p.end() || !it->value().is_array()) {
        throw std::runtime_error("missing/invalid initial_liquidity");
    }
    const auto& arr = it->value().as_array();
    if (arr.size() < 2 || !arr[0].is_string() || !arr[1].is_string()) {
        throw std::runtime_error("initial_liquidity must be [str,str]");
    }

    const std::array<T, 2> amounts = {
        parse_wad<T>(std::string(arr[0].as_string().c_str())),
        parse_wad<T>(std::string(arr[1].as_string().c_str())),
    };

    (void)pool.add_liquidity(amounts, PoolTraits<T>::ZERO(), /*donation=*/false);
    return pool;
}

template <typename T>
json::object run_one_pool(const json::object& pool_obj, const json::object& sequence, uint64_t snapshot_every) {
    const std::string pool_name = get_str(pool_obj, "name");

    bool all_success = true;
    bool last_success = true;
    std::string last_error;

    TwoCryptoPool<T> pool = make_pool_from_json<T>(pool_obj, sequence);

    json::array states;
    json::object last_state;
    bool have_last_state = false;

    if (snapshot_every != 0) {
        states.push_back(snapshot_pool(pool));
    }

    auto it_actions = sequence.find("actions");
    if (it_actions == sequence.end() || !it_actions->value().is_array()) {
        throw std::runtime_error("sequence.actions missing/invalid");
    }

    const auto& actions = it_actions->value().as_array();
    for (size_t action_idx = 0; action_idx < actions.size(); ++action_idx) {
        bool success = true;
        std::string err;

        try {
            const auto& v = actions[action_idx];
            if (!v.is_object()) {
                throw std::runtime_error("action must be object");
            }
            const auto& act = v.as_object();
            const std::string type = get_str(act, "type");

            if (type == "exchange") {
                const int64_t i = act.at("i").as_int64();
                const int64_t j = act.at("j").as_int64();
                const T dx = parse_wad<T>(get_str(act, "dx"));
                (void)pool.exchange(T(i), T(j), dx, PoolTraits<T>::ZERO());
            } else if (type == "add_liquidity") {
                const auto& amts = act.at("amounts").as_array();
                if (amts.size() < 2 || !amts[0].is_string() || !amts[1].is_string()) {
                    throw std::runtime_error("add_liquidity.amounts must be [str,str]");
                }
                const std::array<T, 2> amounts = {
                    parse_wad<T>(std::string(amts[0].as_string().c_str())),
                    parse_wad<T>(std::string(amts[1].as_string().c_str())),
                };
                const bool donation = act.if_contains("donation") ? act.at("donation").as_bool() : false;
                (void)pool.add_liquidity(amounts, PoolTraits<T>::ZERO(), donation);
            } else if (type == "time_travel") {
                if (act.if_contains("seconds")) {
                    const int64_t secs = act.at("seconds").as_int64();
                    if (secs > 0) {
                        pool.advance_time(static_cast<uint64_t>(secs));
                    }
                } else if (act.if_contains("timestamp")) {
                    const uint64_t ts = static_cast<uint64_t>(act.at("timestamp").as_int64());
                    pool.set_block_timestamp(ts);
                }
            }
        } catch (const std::exception& e) {
            success = false;
            err = e.what();
        }

        all_success = all_success && success;
        last_success = success;
        last_error = err;

        if (snapshot_every != 0) {
            json::object st = snapshot_pool(pool);
            st["action_success"] = success;
            if (!success) {
                st["error"] = err;
            }

            if (snapshot_every == 1) {
                states.push_back(st);
            } else {
                if (((action_idx + 1) % snapshot_every) == 0) {
                    states.push_back(st);
                }
                last_state = st;
                have_last_state = true;
            }
        }
    }

    json::object out;
    out["pool_config"] = pool_name;
    out["sequence"] = get_str(sequence, "name");

    json::object res;
    res["success"] = all_success;

    if (snapshot_every == 0) {
        json::object st = snapshot_pool(pool);
        st["action_success"] = last_success;
        if (!last_success) {
            st["error"] = last_error;
        }
        res["final_state"] = st;
    } else {
        if (snapshot_every > 1 && have_last_state) {
            bool same = false;
            if (!states.empty()) {
                const auto& back = states.back();
                if (back.is_object()) {
                    const auto& bo = back.as_object();
                    if (bo.if_contains("timestamp") && last_state.if_contains("timestamp")) {
                        same = bo.at("timestamp") == last_state.at("timestamp");
                    }
                }
            }
            if (!same) {
                states.push_back(last_state);
            }
        }
        res["states"] = states;
    }

    out["result"] = res;
    return out;
}

template <typename T>
int run_harness(const std::string& pools_file, const std::string& sequences_file, const std::string& output_file) {
    const auto pools_val = json::parse(read_file(pools_file));
    const auto seqs_val = json::parse(read_file(sequences_file));

    const auto& pools_obj = pools_val.as_object();
    const auto& seqs_obj = seqs_val.as_object();

    const auto& pools = pools_obj.at("pools").as_array();
    const auto& seqs = seqs_obj.at("sequences").as_array();
    if (seqs.empty()) {
        throw std::runtime_error("No sequences found");
    }
    const auto& sequence = seqs[0].as_object();

    uint64_t snapshot_every = 1;
    if (env_flag("SAVE_LAST_ONLY")) {
        snapshot_every = 0;
    }
    if (const char* v = std::getenv("SNAPSHOT_EVERY")) {
        try {
            const long vv = std::stol(v);
            snapshot_every = (vv <= 0) ? 0 : static_cast<uint64_t>(vv);
        } catch (...) {
        }
    }

    uint64_t threads = std::max<uint64_t>(1, std::thread::hardware_concurrency());
    threads = std::max<uint64_t>(1, env_u64("CPP_THREADS", threads));

    std::atomic<size_t> next{0};
    std::mutex io_mu;
    std::vector<json::object> results(pools.size());

    auto worker = [&]() {
        for (;;) {
            const size_t idx = next.fetch_add(1);
            if (idx >= pools.size()) return;

            const auto& p = pools[idx].as_object();
            const std::string name = get_str(p, "name");

            {
                std::lock_guard<std::mutex> lk(io_mu);
                std::cout << "Processing " << name << "..." << std::endl;
            }

            try {
                results[idx] = run_one_pool<T>(p, sequence, snapshot_every);
            } catch (const std::exception& e) {
                json::object out;
                out["pool_config"] = name;
                out["sequence"] = get_str(sequence, "name");
                json::object res;
                res["success"] = false;
                res["error"] = e.what();
                out["result"] = res;
                results[idx] = out;
            }
        }
    };

    std::vector<std::thread> ws;
    ws.reserve(static_cast<size_t>(threads));
    for (uint64_t t = 0; t < threads; ++t) {
        ws.emplace_back(worker);
    }
    for (auto& th : ws) {
        th.join();
    }

    json::array out_results;
    out_results.reserve(results.size());
    for (auto& r : results) {
        out_results.push_back(r);
    }

    json::object meta;
    meta["pool_configs_file"] = pools_file;
    meta["action_sequences_file"] = sequences_file;
    meta["total_tests"] = static_cast<uint64_t>(pools.size());

    json::object out;
    out["results"] = out_results;
    out["metadata"] = meta;

    std::ofstream of(output_file, std::ios::binary);
    if (!of) {
        throw std::runtime_error("Cannot write output file: " + output_file);
    }
    of << json::serialize(out) << "\n";
    return 0;
}

} // namespace

#if defined(HARNESS_MODE_I)
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <sequences.json> <output.json>\n";
        return 1;
    }
    return run_harness<uint256>(argv[1], argv[2], argv[3]);
}
#elif defined(HARNESS_MODE_D)
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <sequences.json> <output.json>\n";
        return 1;
    }
    return run_harness<double>(argv[1], argv[2], argv[3]);
}
#elif defined(HARNESS_MODE_F)
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <sequences.json> <output.json>\n";
        return 1;
    }
    return run_harness<float>(argv[1], argv[2], argv[3]);
}
#elif defined(HARNESS_MODE_LD)
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <sequences.json> <output.json>\n";
        return 1;
    }
    return run_harness<long double>(argv[1], argv[2], argv[3]);
}
#else
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <mode:i|d|f|ld> <pools.json> <sequences.json> <output.json>\n";
        return 1;
    }

    const std::string mode = argv[1];
    const std::string pools = argv[2];
    const std::string seq = argv[3];
    const std::string out = argv[4];

    if (mode == "i") return run_harness<uint256>(pools, seq, out);
    if (mode == "d") return run_harness<double>(pools, seq, out);
    if (mode == "f") return run_harness<float>(pools, seq, out);
    if (mode == "ld") return run_harness<long double>(pools, seq, out);

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
#endif
