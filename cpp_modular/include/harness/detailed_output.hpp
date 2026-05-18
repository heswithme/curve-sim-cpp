// Detailed per-candle output for compatibility with old simulator format
#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "events/types.hpp"
#include "harness/npz_writer.hpp"

namespace arb {
namespace harness {

// Per-candle state entry matching old simulator's detailed-output.json format
template <typename T>
struct DetailedEntry {
    uint64_t t;           // candle timestamp
    T token0;             // pool balance[0]
    T token1;             // pool balance[1]
    T price_oracle;       // EMA oracle price
    T price_scale;        // price scale
    T profit;             // virtual_price - 1.0
    T vp;                 // virtual price
    T vp_boosted;         // virtual price with donation boost
    T xcp;                // raw xcp_profit value
    T total_supply;       // total LP supply
    T donation_apy;       // annual donation rate used by the harness
    T donation_shares;    // donation shares balance
    T donation_unlocked;  // unlocked donation shares
    T last_prices;        // last spot price used for EMA
    uint64_t last_timestamp; // last EMA update timestamp
    T open;               // candle OHLC
    T high;
    T low;
    T close;
    T p_cex;              // event price used for this tick
    T fee;                // dynamic fee at this point
    uint64_t n_trades;    // cumulative trade count
    uint64_t n_rebalances; // cumulative rebalance count
};

// Write detailed entries to JSON file
// Format: array of objects matching old simulator's detailed-output.json
template <typename T>
bool write_detailed_log(const std::string& path, const std::vector<DetailedEntry<T>>& entries) {
    std::ofstream out(path);
    if (!out) return false;
    
    out << std::setprecision(18);
    out << "[\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        out << "{\"t\": " << e.t
            << ", \"token0\": " << e.token0
            << ", \"token1\": " << e.token1
            << ", \"price_oracle\": " << e.price_oracle
            << ", \"price_scale\": " << e.price_scale
            << ", \"profit\": " << e.profit
            << ", \"vp\": " << e.vp
            << ", \"vp_boosted\": " << e.vp_boosted
            << ", \"xcp\": " << e.xcp
            << ", \"total_supply\": " << e.total_supply
            << ", \"donation_apy\": " << e.donation_apy
            << ", \"donation_shares\": " << e.donation_shares
            << ", \"donation_unlocked\": " << e.donation_unlocked
            << ", \"last_prices\": " << e.last_prices
            << ", \"last_timestamp\": " << e.last_timestamp
            << ", \"open\": " << e.open
            << ", \"high\": " << e.high
            << ", \"low\": " << e.low
            << ", \"close\": " << e.close
            << ", \"p_cex\": " << e.p_cex
            << ", \"fee\": " << e.fee
            << ", \"n_trades\": " << e.n_trades
            << ", \"n_rebalances\": " << e.n_rebalances
            << "}";
        if (i + 1 < entries.size()) out << ",";
        out << "\n";
    }
    out << "]\n";
    
    return out.good();
}

// Write detailed entries to NPZ as one array per field.
// This keeps the same field names as detailed-output.json but avoids giant JSON traces.
template <typename T>
bool write_detailed_npz(const std::string& path, const std::vector<DetailedEntry<T>>& entries) {
    try {
        StoredNpzWriter writer(path);
        std::vector<double> f64;
        std::vector<uint64_t> u64;
        f64.reserve(entries.size());
        u64.reserve(entries.size());

        auto add_f64 = [&](const std::string& name, auto member) {
            f64.clear();
            for (const auto& e : entries) {
                f64.push_back(static_cast<double>(e.*member));
            }
            writer.add_f64(name, f64);
        };
        auto add_u64 = [&](const std::string& name, auto member) {
            u64.clear();
            for (const auto& e : entries) {
                u64.push_back(static_cast<uint64_t>(e.*member));
            }
            writer.add_u64(name, u64);
        };

        add_u64("t", &DetailedEntry<T>::t);
        add_f64("token0", &DetailedEntry<T>::token0);
        add_f64("token1", &DetailedEntry<T>::token1);
        add_f64("price_oracle", &DetailedEntry<T>::price_oracle);
        add_f64("price_scale", &DetailedEntry<T>::price_scale);
        add_f64("profit", &DetailedEntry<T>::profit);
        add_f64("vp", &DetailedEntry<T>::vp);
        add_f64("vp_boosted", &DetailedEntry<T>::vp_boosted);
        add_f64("xcp", &DetailedEntry<T>::xcp);
        add_f64("total_supply", &DetailedEntry<T>::total_supply);
        add_f64("donation_apy", &DetailedEntry<T>::donation_apy);
        add_f64("donation_shares", &DetailedEntry<T>::donation_shares);
        add_f64("donation_unlocked", &DetailedEntry<T>::donation_unlocked);
        add_f64("last_prices", &DetailedEntry<T>::last_prices);
        add_u64("last_timestamp", &DetailedEntry<T>::last_timestamp);
        add_f64("open", &DetailedEntry<T>::open);
        add_f64("high", &DetailedEntry<T>::high);
        add_f64("low", &DetailedEntry<T>::low);
        add_f64("close", &DetailedEntry<T>::close);
        add_f64("p_cex", &DetailedEntry<T>::p_cex);
        add_f64("fee", &DetailedEntry<T>::fee);
        add_u64("n_trades", &DetailedEntry<T>::n_trades);
        add_u64("n_rebalances", &DetailedEntry<T>::n_rebalances);

        writer.close();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace harness
} // namespace arb
