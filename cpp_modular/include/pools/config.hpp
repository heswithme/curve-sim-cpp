// Pool configuration parsing
#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/json.hpp>

#include "core/json_utils.hpp"
#include "trading/costs.hpp"

namespace arb {
namespace pools {

// Pool initialization parameters (floating-point, unit-scaled)
template <typename T>
struct PoolInit {
    std::array<T, 2> precisions{T(1), T(1)};
    T A{T(100000.0)};
    T gamma{T(0)};
    T mid_fee{T(3e-4)};
    T out_fee{T(5e-4)};
    T fee_gamma{T(0.23)};
    T allowed_extra_profit{T(1e-3)};
    T adjustment_step{T(1e-3)};
    T ma_time{T(600.0)};
    T initial_price{T(1.0)};
    std::array<T, 2> initial_liq{T(1e6), T(1e6)};
    uint64_t start_ts{0};
    
    // Donation controls
    T donation_apy{T(0)};
    T donation_frequency{T(0)};  // seconds
    T donation_coins_ratio{T(0.5)};
    
    // Pool identifier (optional)
    std::string tag;
    
    // Echo back original JSON for params block (optional)
    boost::json::object echo_pool{};
    boost::json::object echo_costs{};
};

// JSON parsing helpers live in core/json_utils.hpp

// Parse a single pool entry from JSON object
// Entry format: { "pool": {...}, "costs": {...}, "tag": "..." }
// Or just pool params directly: { "A": ..., "gamma": ..., ... }
template <typename T>
void parse_pool_entry(
    const boost::json::object& entry,
    PoolInit<T>& out_pool,
    arb::trading::Costs<T>& out_costs
) {
    namespace json = boost::json;

    // Extract pool object (may be nested under "pool" key or at top level)
    const json::object& pool = entry.contains("pool") 
        ? entry.at("pool").as_object() 
        : entry;
    
    // Store raw JSON for echo
    out_pool.echo_pool = pool;
    
    // Tag
    if (auto* v = entry.if_contains("tag")) {
        if (v->is_string()) out_pool.tag = v->as_string().c_str();
    }
    
    // Initial liquidity
    if (auto* v = pool.if_contains("initial_liquidity")) {
        const auto& a = v->as_array();
        if (a.size() >= 2) {
            out_pool.initial_liq[0] = parse_scaled_1e18<T>(a[0]);
            out_pool.initial_liq[1] = parse_scaled_1e18<T>(a[1]);
        }
    }
    
    // Pool parameters
    if (auto* v = pool.if_contains("A")) out_pool.A = parse_plain_real<T>(*v);
    if (auto* v = pool.if_contains("gamma")) out_pool.gamma = parse_plain_real<T>(*v);
    if (auto* v = pool.if_contains("mid_fee")) out_pool.mid_fee = parse_fee_1e10<T>(*v);
    if (auto* v = pool.if_contains("out_fee")) out_pool.out_fee = parse_fee_1e10<T>(*v);
    if (auto* v = pool.if_contains("fee_gamma")) out_pool.fee_gamma = parse_scaled_1e18<T>(*v);
    if (auto* v = pool.if_contains("allowed_extra_profit")) out_pool.allowed_extra_profit = parse_scaled_1e18<T>(*v);
    if (auto* v = pool.if_contains("adjustment_step")) out_pool.adjustment_step = parse_scaled_1e18<T>(*v);
    if (auto* v = pool.if_contains("ma_time")) out_pool.ma_time = parse_plain_real<T>(*v);
    if (auto* v = pool.if_contains("initial_price")) out_pool.initial_price = parse_scaled_1e18<T>(*v);
    if (auto* v = pool.if_contains("start_timestamp")) {
        out_pool.start_ts = static_cast<uint64_t>(parse_plain_real<T>(*v));
        if (out_pool.start_ts > 10000000000ULL) {
            out_pool.start_ts /= 1000ULL;
        }
    }
    
    // Donation controls
    if (auto* v = pool.if_contains("donation_apy")) out_pool.donation_apy = parse_plain_real<T>(*v);
    if (auto* v = pool.if_contains("donation_frequency")) out_pool.donation_frequency = parse_plain_real<T>(*v);
    if (auto* v = pool.if_contains("donation_coins_ratio")) {
        T r = parse_plain_real<T>(*v);
        out_pool.donation_coins_ratio = std::clamp<T>(r, T(0), T(1));
    }
    
    // Costs (optional nested object)
    if (auto* c = entry.if_contains("costs")) {
        const auto& co = c->as_object();
        out_pool.echo_costs = co;  // Store raw costs JSON for echo
        if (auto* v = co.if_contains("arb_fee_bps")) out_costs.arb_fee_bps = parse_plain_real<T>(*v);
        if (auto* v = co.if_contains("gas_coin0")) out_costs.gas_coin0 = parse_plain_real<T>(*v);
        if (auto* v = co.if_contains("use_volume_cap")) out_costs.use_volume_cap = v->as_bool();
        if (auto* v = co.if_contains("volume_cap_mult")) out_costs.volume_cap_mult = parse_plain_real<T>(*v);
        if (auto* v = co.if_contains("volume_cap_is_coin_1")) {
            out_costs.volume_cap_is_coin1 = v->is_bool()
                ? v->as_bool()
                : (parse_plain_real<T>(*v) != T(0));
        }
    }
}

// Load all pool entries from a JSON file
// Supports formats:
// - { "pools": [ {...}, {...} ] }
// - { "pool": {...} }
// - [ {...}, {...} ]
// Optional: start_idx/end_idx for loading a subset (0-based, end exclusive)
template <typename T>
std::vector<std::pair<PoolInit<T>, arb::trading::Costs<T>>> 
load_pool_configs(const std::string& path, size_t start_idx = 0, size_t end_idx = SIZE_MAX) {
    namespace json = boost::json;
    
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open pools json: " + path);
    std::ostringstream oss;
    oss << in.rdbuf();
    const std::string s = oss.str();
    
    json::value root = json::parse(s);
    
    std::vector<json::object> entries;
    if (root.is_object()) {
        const auto& obj = root.as_object();
        if (obj.contains("pools")) {
            for (auto& v : obj.at("pools").as_array()) {
                entries.push_back(v.as_object());
            }
        } else if (obj.contains("pool")) {
            entries.push_back(obj);
        } else {
            throw std::runtime_error("Invalid pools json: expected 'pools' array or single 'pool'");
        }
    } else if (root.is_array()) {
        for (auto& v : root.as_array()) {
            entries.push_back(v.as_object());
        }
    } else {
        throw std::runtime_error("Invalid pools json root type");
    }
    
    // Apply range bounds
    if (start_idx >= entries.size()) {
        return {};  // Empty result if start beyond range
    }
    size_t actual_end = std::min(end_idx, entries.size());
    
    std::vector<std::pair<PoolInit<T>, arb::trading::Costs<T>>> result;
    result.reserve(actual_end - start_idx);
    
    for (size_t i = start_idx; i < actual_end; ++i) {
        PoolInit<T> pool_init{};
        arb::trading::Costs<T> costs{};
        parse_pool_entry(entries[i], pool_init, costs);
        result.emplace_back(std::move(pool_init), std::move(costs));
    }
    
    return result;
}

} // namespace pools
} // namespace arb
