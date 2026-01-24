// Detailed per-candle output for compatibility with old simulator format
#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "events/types.hpp"

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

} // namespace harness
} // namespace arb
