// JSON parsing and serialization utilities
#pragma once

#include <boost/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace arb {

// ============================================================================
// Scaling constants
// ============================================================================

constexpr long double WAD = 1e18L;
constexpr long double FEE_SCALE = 1e10L;

// ============================================================================
// Output formatting (value -> string)
// ============================================================================

// Convert a floating-point value to a string representation scaled by 1e18 (wei format)
template <typename T>
inline std::string to_str_1e18(T v) {
    long double scaled = static_cast<long double>(v) * WAD;
    if (!std::isfinite(scaled)) scaled = 0;
    if (scaled < 0) scaled = 0;
    const auto rounded = std::floor(scaled + 0.5L);
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(0);
    oss << rounded;
    return oss.str();
}

// Convert a floating-point value to a rounded integer string (no scaling)
template <typename T>
inline std::string to_int_string(T v) {
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

// ============================================================================
// JSON value parsing (boost::json::value -> T)
// ============================================================================

// Parse a JSON value as a plain real number (no scaling)
template <typename T>
inline T parse_plain_real(const boost::json::value& v) {
    if (v.is_string()) return static_cast<T>(std::strtold(v.as_string().c_str(), nullptr));
    if (v.is_double()) return static_cast<T>(v.as_double());
    if (v.is_int64())  return static_cast<T>(v.as_int64());
    if (v.is_uint64()) return static_cast<T>(v.as_uint64());
    return T(0);
}

// Parse a JSON value as a real number, scaling down from 1e18 representation
template <typename T>
inline T parse_scaled_1e18(const boost::json::value& v) {
    if (v.is_string()) return static_cast<T>(std::strtold(v.as_string().c_str(), nullptr) / WAD);
    if (v.is_double()) return static_cast<T>(v.as_double() / WAD);
    if (v.is_int64())  return static_cast<T>(static_cast<long double>(v.as_int64()) / WAD);
    if (v.is_uint64()) return static_cast<T>(static_cast<long double>(v.as_uint64()) / WAD);
    return T(0);
}

// Parse a JSON value as a fee (scaled down from 1e10)
template <typename T>
inline T parse_fee_1e10(const boost::json::value& v) {
    if (v.is_string()) return static_cast<T>(std::strtold(v.as_string().c_str(), nullptr) / FEE_SCALE);
    if (v.is_double()) return static_cast<T>(v.as_double() / FEE_SCALE);
    if (v.is_int64())  return static_cast<T>(static_cast<long double>(v.as_int64()) / FEE_SCALE);
    if (v.is_uint64()) return static_cast<T>(static_cast<long double>(v.as_uint64()) / FEE_SCALE);
    return T(0);
}

// ============================================================================
// JSON object accessors
// ============================================================================

// Get a required string value from a JSON object (throws on missing/wrong type)
inline std::string get_str(const boost::json::object& obj, const char* key) {
    auto it = obj.find(key);
    if (it == obj.end()) {
        throw std::runtime_error(std::string("missing key: ") + key);
    }
    if (!it->value().is_string()) {
        throw std::runtime_error(std::string("expected string for key: ") + key);
    }
    return std::string(it->value().as_string().c_str());
}

// Get an optional uint64 value from a JSON object (returns default if missing)
inline uint64_t get_u64_opt(const boost::json::object& obj, const char* key, uint64_t default_value) {
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

// ============================================================================
// Environment variable helpers
// ============================================================================

// Get an environment variable as uint64_t (returns default if not set or invalid)
inline uint64_t env_u64(const char* key, uint64_t default_value) {
    if (const char* v = std::getenv(key)) {
        try {
            return static_cast<uint64_t>(std::stoull(v));
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

// Check if an environment variable is set to "1"
inline bool env_flag(const char* key) {
    if (const char* v = std::getenv(key)) {
        return std::string(v) == "1";
    }
    return false;
}

// ============================================================================
// File I/O
// ============================================================================

// Read entire file contents into a string
inline std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

} // namespace arb
