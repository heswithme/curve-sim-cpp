#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "pools/twocrypto_fx/stableswap_math.hpp"

namespace {

using U256 = arb::pools::twocrypto_fx::uint256;
using Ops  = arb::pools::twocrypto_fx::MathOps<U256>;

U256 parse_u256(const char* s) {
    if (!s) return U256(0);
    return U256(s);
}

char* dup_cstr(const std::string& s) {
    const size_t n = s.size();
    char* out = static_cast<char*>(std::malloc(n + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.data(), n);
    out[n] = '\0';
    return out;
}

std::string to_string_u256(const U256& x) {
    return x.convert_to<std::string>();
}

} // namespace

extern "C" {

// Matches historical Python ctypes ABI used by python/benchmark_math/main.py

char* newton_D(const char* A, const char* gamma, const char* x0, const char* x1) {
    try {
        const U256 amp = parse_u256(A);
        const U256 gam = parse_u256(gamma);
        const auto xp = std::array<U256, 2>{parse_u256(x0), parse_u256(x1)};

        const U256 D = Ops::newton_D(amp, gam, xp, U256(0));
        return dup_cstr(to_string_u256(D));
    } catch (...) {
        return dup_cstr("0");
    }
}

char** get_y(const char* A, const char* gamma, const char* x0, const char* x1, const char* D, int i) {
    try {
        const U256 amp = parse_u256(A);
        const U256 gam = parse_u256(gamma);
        const auto xp  = std::array<U256, 2>{parse_u256(x0), parse_u256(x1)};
        const U256 inv = parse_u256(D);

        const auto res = Ops::get_y(amp, gam, xp, inv, static_cast<size_t>(i));

        char** out = static_cast<char**>(std::malloc(sizeof(char*) * 2));
        if (!out) return nullptr;
        out[0] = dup_cstr(to_string_u256(res.value));
        out[1] = dup_cstr("0");
        return out;
    } catch (...) {
        char** out = static_cast<char**>(std::malloc(sizeof(char*) * 2));
        if (!out) return nullptr;
        out[0] = dup_cstr("0");
        out[1] = dup_cstr("0");
        return out;
    }
}

char* get_p(const char* x0, const char* x1, const char* D, const char* A) {
    try {
        const auto xp = std::array<U256, 2>{parse_u256(x0), parse_u256(x1)};
        const U256 inv = parse_u256(D);
        const auto A_gamma = std::array<U256, 2>{parse_u256(A), parse_u256("145000000000000")};

        const U256 p = Ops::get_p(xp, inv, A_gamma);
        return dup_cstr(to_string_u256(p));
    } catch (...) {
        return dup_cstr("0");
    }
}

void free_string(void* p) {
    std::free(p);
}

void free_string_array(char** arr, int n) {
    if (!arr) return;
    for (int i = 0; i < n; ++i) {
        std::free(arr[i]);
    }
    std::free(arr);
}

} // extern "C"
