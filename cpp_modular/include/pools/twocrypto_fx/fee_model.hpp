#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string_view>
#include <type_traits>

#include "stableswap_math.hpp"

namespace arb {
namespace pools {
namespace twocrypto_fx {

inline constexpr std::size_t FEE_PARAM_COUNT = 20;
inline constexpr std::size_t FEE_STATE_COUNT = 20;
inline constexpr std::size_t FEE_COMPONENT_COUNT = 4;
inline constexpr std::size_t FEE_SIGNAL_COUNT = 4;
inline constexpr std::string_view FEE_MODEL_NAME = "constant_fee_v1";
inline constexpr std::array<std::string_view, FEE_PARAM_COUNT> FEE_PARAM_LABELS{{
    "fee_value",
    "reserved_1",
    "reserved_2",
    "reserved_3",
    "reserved_4",
    "reserved_5",
    "reserved_6",
    "reserved_7",
    "reserved_8",
    "reserved_9",
    "reserved_10",
    "reserved_11",
    "reserved_12",
    "reserved_13",
    "reserved_14",
    "reserved_15",
    "reserved_16",
    "reserved_17",
    "reserved_18",
    "reserved_19",
}};

template <typename T>
using FeeParams = std::array<T, FEE_PARAM_COUNT>;

template <typename T>
using FeeState = std::array<T, FEE_STATE_COUNT>;

template <typename T>
struct FeeInputs {
    std::array<T, 2> xp{};
    T spot{0};
    T price_oracle{0};
    T price_scale{0};
    T last_prices{0};
    std::size_t idx_i{0};
    std::size_t idx_j{0};
    T dx{0};
    T cex_price{0};
    bool has_trade_context{false};
    uint64_t block_timestamp{0};
    uint64_t last_timestamp{0};
    T precision{1};
    T fee_precision{1};
};

template <typename T>
struct FeeStateInputs {
    T spot{0};
    T price_oracle{0};
    T price_scale{0};
    T last_prices{0};
    uint64_t block_timestamp{0};
    T ma_time{0};
    T precision{1};
};

template <typename T>
struct FeeBreakdown {
    T total_fee{0};
    std::array<T, FEE_COMPONENT_COUNT> components{};
    std::array<T, FEE_SIGNAL_COUNT> signals{};
};

template <typename T>
struct ExchangePreview {
    T dy_after_fee{0};
    T fee_tokens{0};
    T spot_post{0};
    FeeBreakdown<T> fees{};
};

enum FeeParamIndex : std::size_t {
    FEE_PARAM_FEE_VALUE = 0,
    FEE_PARAM_RESERVED_1 = 1,
    FEE_PARAM_RESERVED_2 = 2,
    FEE_PARAM_RESERVED_3 = 3,
    FEE_PARAM_RESERVED_4 = 4,
    FEE_PARAM_RESERVED_5 = 5,
    FEE_PARAM_RESERVED_6 = 6,
    FEE_PARAM_RESERVED_7 = 7,
    FEE_PARAM_RESERVED_8 = 8,
    FEE_PARAM_RESERVED_9 = 9,
    FEE_PARAM_RESERVED_10 = 10,
    FEE_PARAM_RESERVED_11 = 11,
    FEE_PARAM_RESERVED_12 = 12,
    FEE_PARAM_RESERVED_13 = 13,
    FEE_PARAM_RESERVED_14 = 14,
    FEE_PARAM_RESERVED_15 = 15,
    FEE_PARAM_RESERVED_16 = 16,
    FEE_PARAM_RESERVED_17 = 17,
    FEE_PARAM_RESERVED_18 = 18,
    FEE_PARAM_RESERVED_19 = 19,
};

enum FeeStateIndex : std::size_t {
    FEE_STATE_RESERVED_0 = 0,
    FEE_STATE_RESERVED_1 = 1,
    FEE_STATE_RESERVED_2 = 2,
    FEE_STATE_RESERVED_3 = 3,
    FEE_STATE_RESERVED_4 = 4,
    FEE_STATE_RESERVED_5 = 5,
    FEE_STATE_RESERVED_6 = 6,
    FEE_STATE_RESERVED_7 = 7,
    FEE_STATE_RESERVED_8 = 8,
    FEE_STATE_RESERVED_9 = 9,
    FEE_STATE_RESERVED_10 = 10,
    FEE_STATE_RESERVED_11 = 11,
    FEE_STATE_RESERVED_12 = 12,
    FEE_STATE_RESERVED_13 = 13,
    FEE_STATE_RESERVED_14 = 14,
    FEE_STATE_RESERVED_15 = 15,
    FEE_STATE_RESERVED_16 = 16,
    FEE_STATE_RESERVED_17 = 17,
    FEE_STATE_RESERVED_18 = 18,
    FEE_STATE_RESERVED_19 = 19,
};

enum FeeComponentIndex : std::size_t {
    FEE_COMPONENT_CONSTANT = 0,
    FEE_COMPONENT_RESERVED_1 = 1,
    FEE_COMPONENT_RESERVED_2 = 2,
    FEE_COMPONENT_RESERVED = 3,
};

enum FeeSignalIndex : std::size_t {
    FEE_SIGNAL_RESERVED_0 = 0,
    FEE_SIGNAL_RESERVED_1 = 1,
    FEE_SIGNAL_RESERVED_2 = 2,
    FEE_SIGNAL_RESERVED = 3,
};

template <typename T>
inline FeeParams<T> zero_fee_params() {
    return FeeParams<T>{};
}

template <typename T>
inline FeeState<T> zero_fee_state() {
    return FeeState<T>{};
}

template <typename T>
inline FeeParams<T> make_constant_fee_params(const T& fee_value) {
    FeeParams<T> params{};
    params[FEE_PARAM_FEE_VALUE] = fee_value;
    return params;
}

template <typename T>
inline T constant_fee_from_params(const FeeParams<T>& fee_params) {
    return fee_params[FEE_PARAM_FEE_VALUE];
}

template <typename T>
inline void init_fee_state(
    FeeState<T>& fee_state,
    const T& spot,
    uint64_t block_timestamp
) {
    (void)spot;
    (void)block_timestamp;
    fee_state = zero_fee_state<T>();
}

template <typename T>
inline void step_fee_state(
    const FeeParams<T>& fee_params,
    FeeState<T>& fee_state,
    const FeeStateInputs<T>& in
) {
    (void)fee_params;
    (void)fee_state;
    (void)in;
}

template <typename T>
inline FeeBreakdown<T> eval_fee_model(
    const FeeParams<T>& fee_params,
    const FeeState<T>& fee_state,
    const FeeInputs<T>& in
) {
    FeeBreakdown<T> out;
    (void)fee_state;
    (void)in;

    const T fee_value = constant_fee_from_params(fee_params);
    out.components[FEE_COMPONENT_CONSTANT] = fee_value;
    out.total_fee = fee_value;
    if (out.total_fee < T(0)) {
        out.total_fee = T(0);
    }
    if (out.total_fee > in.fee_precision) {
        out.total_fee = in.fee_precision;
    }
    return out;
}

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
