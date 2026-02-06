// Events module - data types for candles and price events
#pragma once

#include <cstdint>

namespace arb {

// CEX candle data (OHLCV)
struct Candle {
    uint64_t ts;
    double open;
    double high;
    double low;
    double close;
    double volume;
};

// Simplified price event (timestamp + price + volume + source candle index)
struct Event {
    uint64_t ts;
    double p_cex;
    double volume;
    uint32_t candle_idx;  // index into candle vector (used for detailed logging)
};

} // namespace arb
