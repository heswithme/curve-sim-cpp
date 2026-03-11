// Runtime-selectable pool backend for arb_harness and eval_server
#pragma once

#include <string>

namespace arb {
namespace harness {

enum class PoolBackend {
    Double,
    LongDouble,
    Uint,
};

inline const char* pool_backend_name(PoolBackend backend) {
    switch (backend) {
        case PoolBackend::Double:
            return "double";
        case PoolBackend::LongDouble:
            return "ld";
        case PoolBackend::Uint:
            return "uint";
    }
    return "double";
}

inline bool parse_pool_backend(const std::string& value, PoolBackend& out_backend) {
    if (value.empty() || value == "double" || value == "d") {
        out_backend = PoolBackend::Double;
        return true;
    }
    if (value == "ld" || value == "long-double" || value == "long_double") {
        out_backend = PoolBackend::LongDouble;
        return true;
    }
    if (value == "uint" || value == "u256" || value == "uint256") {
        out_backend = PoolBackend::Uint;
        return true;
    }
    return false;
}

} // namespace harness
} // namespace arb
