// Cowswap organic trade replay
// Loads historical cowswap trades and replays them if pool offers better price
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pools/twocrypto_fx/helpers.hpp"

namespace arb {
namespace trading {

// Debug flag - set COWSWAP_DEBUG=1 to enable verbose output
inline bool cowswap_debug_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("COWSWAP_DEBUG");
        return env && std::string(env) == "1";
    }();
    return enabled;
}

// Single cowswap trade from historical data
struct CowswapTrade {
    uint64_t ts;          // Unix timestamp
    bool is_buy;          // true = BUY (user buys WBTC with USD, coin0->coin1)
                          // false = SELL (user sells WBTC for USD, coin1->coin0)
    double usd_amount;    // coin0 amount
    double wbtc_amount;   // coin1 amount
    double price;         // Historical execution price
};

// Load cowswap trades from CSV file
// CSV format: unix_timestamp,date,direction,wbtc_amount,usd_amount,price,total_usd_volume,source
// CSV is typically sorted newest-first, we reverse to ascending order
inline std::vector<CowswapTrade> load_cowswap_csv(const std::string& path) {
    std::vector<CowswapTrade> trades;
    std::ifstream file(path);
    if (!file) {
        return trades;  // Return empty on error
    }
    
    std::string line;
    // Skip header
    if (!std::getline(file, line)) {
        return trades;
    }
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Parse CSV: unix_timestamp,date,direction,wbtc_amount,usd_amount,price,...
        std::istringstream ss(line);
        std::string token;
        
        CowswapTrade trade{};
        int col = 0;
        
        while (std::getline(ss, token, ',')) {
            switch (col) {
                case 0:  // unix_timestamp
                    trade.ts = static_cast<uint64_t>(std::stoull(token));
                    break;
                case 1:  // date - skip
                    break;
                case 2:  // direction
                    trade.is_buy = (token == "BUY");
                    break;
                case 3:  // wbtc_amount
                    trade.wbtc_amount = std::stod(token);
                    break;
                case 4:  // usd_amount
                    trade.usd_amount = std::stod(token);
                    break;
                case 5:  // price
                    trade.price = std::stod(token);
                    break;
                default:
                    break;
            }
            col++;
        }
        
        if (col >= 6 && trade.ts > 0) {
            trades.push_back(trade);
        }
    }
    
    // Sort by timestamp ascending (CSV is typically newest-first)
    std::sort(trades.begin(), trades.end(),
              [](const CowswapTrade& a, const CowswapTrade& b) {
                  return a.ts < b.ts;
              });
    
    return trades;
}

// Metrics for cowswap trading (accumulated per pool run)
template <typename T>
struct CowswapMetrics {
    size_t trades_executed{0};
    size_t trades_skipped{0};
    T notional_coin0{0};      // Total notional executed in coin0 terms
    T lp_fee_coin0{0};        // LP fees paid in coin0 terms
};

// Self-contained cowswap trader that manages its own cursor
// Can own trades data or reference shared data
template <typename T>
class CowswapTrader {
public:
    CowswapTrader() = default;
    
    // Construct from loaded trades (takes ownership via move)
    explicit CowswapTrader(std::vector<CowswapTrade>&& trades)
        : owned_trades_(std::move(trades))
        , trades_(&owned_trades_)
        , idx_(0) {}
    
    // Construct from shared trades pointer (non-owning reference)
    // Each instance has independent cursor but shares trade data
    explicit CowswapTrader(const std::vector<CowswapTrade>* trades)
        : trades_(trades)
        , idx_(0) {}
    
    // Load from CSV file directly (owning)
    static CowswapTrader from_csv(const std::string& path) {
        return CowswapTrader(load_cowswap_csv(path));
    }
    
    // Initialize cursor to start at or after given timestamp
    // Call this once after construction, before processing events
    void init_at(uint64_t start_ts) {
        if (!trades_ || trades_->empty()) {
            idx_ = 0;
            return;
        }
        // Binary search for first trade with ts >= start_ts
        auto it = std::lower_bound(trades_->begin(), trades_->end(), start_ts,
            [](const CowswapTrade& t, uint64_t ts) { return t.ts < ts; });
        idx_ = static_cast<size_t>(it - trades_->begin());
    }
    
    // Check if trader is enabled (has trades loaded)
    bool enabled() const {
        return trades_ != nullptr && !trades_->empty();
    }
    
    // Check if there are remaining trades to process
    bool has_pending() const {
        return trades_ != nullptr && idx_ < trades_->size();
    }
    
    // Total number of trades loaded
    size_t total_trades() const {
        return trades_ ? trades_->size() : 0;
    }
    
    // Number of trades already processed (executed + skipped)
    size_t processed_count() const {
        return idx_;
    }
    
    // Apply any trades whose timestamp <= pool timestamp
    // Advances internal cursor automatically
    // Returns number of trades executed
    template <typename Pool>
    size_t apply_due_trades(Pool& pool, CowswapMetrics<T>& metrics) {
        if (!trades_ || idx_ >= trades_->size()) {
            return 0;
        }
        
        size_t executed = 0;
        const uint64_t pool_ts = pool.block_timestamp;
        
        while (idx_ < trades_->size() && (*trades_)[idx_].ts <= pool_ts) {
            const auto& trade = (*trades_)[idx_];
            ++idx_;
            
            // Token mapping:
            // coin0 = USD (quote), coin1 = WBTC (base)
            // BUY: user spends USD (coin0) to get WBTC (coin1)
            //      Execute if pool output (dy) >= historical wbtc_amount
            // SELL: user spends WBTC (coin1) to get USD (coin0)
            //       Execute if pool output (dy) >= historical usd_amount
            
            if (trade.is_buy) {
                // BUY: coin0 -> coin1
                const T dx = static_cast<T>(trade.usd_amount);
                const T required_dy = static_cast<T>(trade.wbtc_amount);
                
                // Simulate to check if we beat historical price
                auto [sim_dy, sim_fee] = pools::twocrypto_fx::simulate_exchange_once(
                    pool, 0, 1, dx);
                
                // Historical effective price: usd_amount / wbtc_amount (USD per BTC)
                // Pool effective price: dx / sim_dy (USD per BTC)
                const T hist_price = dx / required_dy;
                const T pool_price = (sim_dy > T(0)) ? dx / sim_dy : T(0);
                
                if (cowswap_debug_enabled()) {
                    std::cerr << "[COWSWAP] ts=" << trade.ts 
                              << " BUY dx_usd=" << std::fixed << std::setprecision(2) << static_cast<double>(dx)
                              << " hist_btc=" << std::setprecision(8) << static_cast<double>(required_dy)
                              << " pool_btc=" << static_cast<double>(sim_dy)
                              << " hist_price=" << std::setprecision(2) << static_cast<double>(hist_price)
                              << " pool_price=" << static_cast<double>(pool_price)
                              << " ps=" << std::setprecision(0) << static_cast<double>(pool.cached_price_scale)
                              << " => " << (sim_dy >= required_dy ? "EXEC" : "SKIP") << "\n";
                }
                
                if (sim_dy >= required_dy) {
                    // Execute the trade
                    try {
                        auto res = pool.exchange(T(0), T(1), dx, T(0));
                        const T fee_tokens = res[1];
                        
                        // Update metrics
                        metrics.trades_executed++;
                        metrics.notional_coin0 += dx;
                        // Fee is in coin1 (WBTC), convert to coin0 using pool price
                        metrics.lp_fee_coin0 += fee_tokens * pool.cached_price_scale;
                        
                        ++executed;
                    } catch (...) {
                        // Trade failed, count as skipped
                        metrics.trades_skipped++;
                    }
                } else {
                    metrics.trades_skipped++;
                }
            } else {
                // SELL: coin1 -> coin0
                const T dx = static_cast<T>(trade.wbtc_amount);
                const T required_dy = static_cast<T>(trade.usd_amount);
                
                // Simulate to check if we beat historical price
                auto [sim_dy, sim_fee] = pools::twocrypto_fx::simulate_exchange_once(
                    pool, 1, 0, dx);
                
                // Historical effective price: usd_amount / wbtc_amount (USD per BTC)
                // Pool effective price: sim_dy / dx (USD per BTC)
                const T hist_price = required_dy / dx;
                const T pool_price = sim_dy / dx;
                
                if (cowswap_debug_enabled()) {
                    std::cerr << "[COWSWAP] ts=" << trade.ts
                              << " SELL dx_btc=" << std::fixed << std::setprecision(8) << static_cast<double>(dx)
                              << " hist_usd=" << std::setprecision(2) << static_cast<double>(required_dy)
                              << " pool_usd=" << static_cast<double>(sim_dy)
                              << " hist_price=" << static_cast<double>(hist_price)
                              << " pool_price=" << static_cast<double>(pool_price)
                              << " ps=" << std::setprecision(0) << static_cast<double>(pool.cached_price_scale)
                              << " => " << (sim_dy >= required_dy ? "EXEC" : "SKIP") << "\n";
                }
                
                if (sim_dy >= required_dy) {
                    // Execute the trade
                    try {
                        auto res = pool.exchange(T(1), T(0), dx, T(0));
                        const T fee_tokens = res[1];
                        
                        // Update metrics
                        metrics.trades_executed++;
                        // Notional in coin0: dx * price_scale
                        metrics.notional_coin0 += dx * pool.cached_price_scale;
                        // Fee is in coin0 (USD)
                        metrics.lp_fee_coin0 += fee_tokens;
                        
                        ++executed;
                    } catch (...) {
                        metrics.trades_skipped++;
                    }
                } else {
                    metrics.trades_skipped++;
                }
            }
        }
        
        return executed;
    }
    
private:
    std::vector<CowswapTrade> owned_trades_;       // Storage if owning
    const std::vector<CowswapTrade>* trades_{nullptr};  // Pointer to trades (owned or shared)
    size_t idx_{0};
};

} // namespace trading
} // namespace arb
