// Events module - loading and generation functions
#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace arb {

// Load candles from JSON file.
// Format: array of [ts, open, high, low, close, volume]
// squeeze_frac: clamp high/low to ±squeeze_frac around (open+close)/2
std::vector<Candle> load_candles(const std::string& path,
                                  size_t max_candles = 0,
                                  double squeeze_frac = 0.999);

// Generate two price events per candle (low-first or high-first path).
std::vector<Event> gen_events(const std::vector<Candle>& candles);

} // namespace arb
