// Events module - implementation (non-templated)
#include "events/loader.hpp"

#include <boost/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace json = boost::json;

namespace arb {

std::vector<Candle> load_candles(const std::string& path,
                                  size_t max_candles,
                                  double squeeze_frac) {
    std::vector<Candle> out;
    out.reserve(1024);

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open candles file: " + path);

    std::ostringstream oss;
    oss << in.rdbuf();
    const std::string s = oss.str();

    json::value val = json::parse(s);
    if (!val.is_array()) throw std::runtime_error("Candles JSON must be an array of arrays");

    const auto& arr = val.as_array();
    const size_t limit = (max_candles ? std::min(max_candles, arr.size()) : arr.size());
    out.reserve(limit);

    auto to_d = [](const json::value& v) -> double {
        if (v.is_double()) return v.as_double();
        if (v.is_int64())  return static_cast<double>(v.as_int64());
        if (v.is_uint64()) return static_cast<double>(v.as_uint64());
        return 0.0;
    };

    for (size_t idx = 0; idx < limit; ++idx) {
        const auto& a = arr[idx].as_array();
        if (a.size() < 6) continue;

        Candle c{};
        uint64_t ts = 0;
        const auto& tsv = a[0];
        if (tsv.is_uint64()) ts = tsv.as_uint64();
        else if (tsv.is_int64()) ts = static_cast<uint64_t>(tsv.as_int64());
        else if (tsv.is_double()) ts = static_cast<uint64_t>(tsv.as_double());
        if (ts > 10000000000ULL) ts /= 1000ULL; // ms->s
        c.ts = ts;

        c.open   = to_d(a[1]);
        c.high   = to_d(a[2]);
        c.low    = to_d(a[3]);
        c.close  = to_d(a[4]);
        c.volume = to_d(a[5]);

        // Apply squeeze filter
        if (squeeze_frac > 0.0) {
            const double oc_mid = 0.5 * (c.open + c.close);
            if (oc_mid > 0) {
                const double max_h = oc_mid * (1.0 + squeeze_frac);
                const double min_l = oc_mid * (1.0 - squeeze_frac);
                if (c.high > max_h) c.high = max_h;
                if (c.low  < min_l) c.low  = min_l;
            }
        }
        out.push_back(c);
    }
    
    // Sort candles by timestamp (input may not be ordered)
    std::sort(out.begin(), out.end(), [](const Candle& a, const Candle& b) {
        return a.ts < b.ts;
    });
    
    return out;
}

std::vector<Event> gen_events(const std::vector<Candle>& cs) {
    std::vector<Event> evs;
    evs.reserve(cs.size() * 2);

    for (const auto& c : cs) {
        // Choose path: "low first" vs "high first" based on which is shorter
        const double path1 = std::abs(c.open - c.low)  + std::abs(c.high - c.close);
        const double path2 = std::abs(c.open - c.high) + std::abs(c.low  - c.close);
        const bool first_low = path1 < path2;

        evs.push_back(Event{c.ts,      first_low ? c.low  : c.high, c.volume / 2.0, c});
        evs.push_back(Event{c.ts + 10, first_low ? c.high : c.low,  c.volume / 2.0, c});
    }

    std::sort(evs.begin(), evs.end(), [](const Event& a, const Event& b) {
        return a.ts < b.ts;
    });
    return evs;
}

} // namespace arb
