// Conversion helpers for mixed harness/pool numeric modes.
#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <type_traits>

#include "pools/twocrypto_fx/stableswap_math.hpp"

namespace arb {
namespace harness {

namespace fx = arb::pools::twocrypto_fx;

template <typename T>
struct PoolValue {
    template <typename H>
    static T amount_from_human(H v) { return static_cast<T>(v); }

    template <typename H>
    static T amount_from_human_floor(H v) { return static_cast<T>(v); }

    template <typename H>
    static T price_from_human(H v) { return static_cast<T>(v); }

    template <typename H>
    static T fee_from_human(H v) { return static_cast<T>(v); }

    template <typename H>
    static H amount_to_human(const T& v) { return static_cast<H>(v); }

    template <typename H>
    static H price_to_human(const T& v) { return static_cast<H>(v); }

    template <typename H>
    static H fee_to_human(const T& v) { return static_cast<H>(v); }

};

template <>
struct PoolValue<fx::uint256> {
    using U = fx::uint256;

    static U wad() {
        static U v("1000000000000000000");
        return v;
    }

    static U fee_scale() {
        static U v("10000000000");
        return v;
    }

    template <typename H>
    static U scaled_floor(H v, const U& scale) {
        const long double x = static_cast<long double>(v);
        if (!(x > 0.0L) || !std::isfinite(x)) {
            return U(0);
        }
        const long double scaled = std::floor(x * scale.convert_to<long double>());
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(0);
        oss << scaled;
        return U(oss.str());
    }

    template <typename H>
    static U amount_from_human(H v) { return scaled_floor(v, wad()); }

    template <typename H>
    static U amount_from_human_floor(H v) { return scaled_floor(v, wad()); }

    template <typename H>
    static U price_from_human(H v) { return scaled_floor(v, wad()); }

    template <typename H>
    static U fee_from_human(H v) { return scaled_floor(v, fee_scale()); }

    template <typename H>
    static H amount_to_human(const U& v) {
        return static_cast<H>(v.convert_to<long double>() / 1.0e18L);
    }

    template <typename H>
    static H price_to_human(const U& v) {
        return static_cast<H>(v.convert_to<long double>() / 1.0e18L);
    }

    template <typename H>
    static H fee_to_human(const U& v) {
        return static_cast<H>(v.convert_to<long double>() / 1.0e10L);
    }

};

template <typename H, typename P>
inline H pool_amount_to_h(const P& v) {
    return PoolValue<P>::template amount_to_human<H>(v);
}

template <typename H, typename P>
inline H pool_price_to_h(const P& v) {
    return PoolValue<P>::template price_to_human<H>(v);
}

template <typename H, typename P>
inline H pool_fee_to_h(const P& v) {
    return PoolValue<P>::template fee_to_human<H>(v);
}

template <typename P, typename H>
inline P h_amount_to_pool(H v) {
    return PoolValue<P>::amount_from_human(v);
}

template <typename P, typename H>
inline P h_amount_to_pool_floor(H v) {
    return PoolValue<P>::amount_from_human_floor(v);
}

template <typename P, typename H>
inline P h_price_to_pool(H v) {
    return PoolValue<P>::price_from_human(v);
}

template <typename P, typename H>
inline P h_fee_to_pool(H v) {
    return PoolValue<P>::fee_from_human(v);
}

template <typename P>
inline constexpr bool is_uint_pool_v = std::is_same_v<P, fx::uint256>;

} // namespace harness
} // namespace arb
