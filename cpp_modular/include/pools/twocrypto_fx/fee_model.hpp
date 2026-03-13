#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>

#include "stableswap_math.hpp"

namespace arb {
namespace pools {
namespace twocrypto_fx {

inline constexpr std::size_t FEE_PARAM_COUNT = 20;
inline constexpr std::size_t FEE_STATE_COUNT = 20;
inline constexpr std::size_t FEE_COMPONENT_COUNT = 4;
inline constexpr std::size_t FEE_SIGNAL_COUNT = 4;

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
    FEE_PARAM_BASE_FLOOR = 0,
    FEE_PARAM_MID_FEE = 1,
    FEE_PARAM_SPREAD = 2,
    FEE_PARAM_FEE_GAMMA = 3,
    FEE_PARAM_CALM_DISCOUNT_MAX = 4,
    FEE_PARAM_FEE_VOLATILITY_REF = 5,
    FEE_PARAM_GAP_FEE_SCALE = 6,
    FEE_PARAM_GAP_FEE_CONST_DISCOUNT = 7,
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
    FEE_STATE_VOL_EMA = 0,
    FEE_STATE_LAST_SPOT = 1,
    FEE_STATE_LAST_UPDATE_TS = 2,
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
    FEE_COMPONENT_BASE = 0,
    FEE_COMPONENT_INVENTORY = 1,
    FEE_COMPONENT_GAP = 2,
    FEE_COMPONENT_RESERVED = 3,
};

enum FeeSignalIndex : std::size_t {
    FEE_SIGNAL_BALANCE = 0,
    FEE_SIGNAL_VOLATILITY = 1,
    FEE_SIGNAL_DIRECTIONAL_GAP = 2,
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
inline FeeParams<T> make_current_fee_params(
    const T& base_floor,
    const T& mid_fee,
    const T& out_fee,
    const T& fee_gamma,
    const T& calm_discount_max,
    const T& fee_volatility_ref,
    const T& gap_fee_scale,
    const T& gap_fee_const_discount
) {
    FeeParams<T> params{};
    params[FEE_PARAM_BASE_FLOOR] = base_floor;
    params[FEE_PARAM_MID_FEE] = mid_fee;
    params[FEE_PARAM_SPREAD] = (out_fee > mid_fee) ? (out_fee - mid_fee) : T(0);
    params[FEE_PARAM_FEE_GAMMA] = fee_gamma;
    params[FEE_PARAM_CALM_DISCOUNT_MAX] = calm_discount_max;
    params[FEE_PARAM_FEE_VOLATILITY_REF] = fee_volatility_ref;
    params[FEE_PARAM_GAP_FEE_SCALE] = gap_fee_scale;
    params[FEE_PARAM_GAP_FEE_CONST_DISCOUNT] = gap_fee_const_discount;
    return params;
}

template <typename T>
inline T fee_max(const T& a, const T& b) {
    return (a > b) ? a : b;
}

template <typename T>
inline T fee_abs_diff(const T& a, const T& b) {
    return (a >= b) ? (a - b) : (b - a);
}

template <typename T>
inline T scaled_stress_score(const T& value, const T& ref, const T& precision) {
    if (!(value > T(0)) || !(ref > T(0))) {
        return T(0);
    }
    const T scaled = (value * precision) / ref;
    return (scaled < precision) ? scaled : precision;
}

template <typename T>
inline T relative_gap(const T& lhs, const T& rhs, const T& precision) {
    if (!(lhs > T(0)) || !(rhs > T(0))) {
        return T(0);
    }
    const T scaled = (lhs * precision) / rhs;
    return fee_abs_diff(scaled, precision);
}

template <typename T>
inline T balance_indicator(const std::array<T, 2>& xp) {
    const T sum = xp[0] + xp[1];
    if (!(sum > T(0))) {
        return T(0);
    }
    return T(4) * xp[0] * xp[1] / (sum * sum);
}

template <typename T>
inline T inventory_fee_from_params(
    const std::array<T, 2>& xp,
    const FeeParams<T>& fee_params,
    const T& precision
) {
    const T mid_fee = fee_params[FEE_PARAM_MID_FEE];
    const T spread = fee_params[FEE_PARAM_SPREAD];
    const T out_fee = mid_fee + spread;
    const T fee_gamma = fee_params[FEE_PARAM_FEE_GAMMA];

    if (!(fee_gamma > T(0))) {
        return mid_fee;
    }

    const T sum = xp[0] + xp[1];
    if (!(sum > T(0))) {
        return mid_fee;
    }

    T B = precision * T(4) * xp[0] / sum * xp[1] / sum;
    B = fee_gamma * B / (fee_gamma * B / precision + precision - B);
    return (mid_fee * B + out_fee * (precision - B)) / precision;
}

template <typename T>
inline T fee_state_vol_ema(const FeeState<T>& fee_state) {
    return fee_state[FEE_STATE_VOL_EMA];
}

template <typename T>
inline T fee_state_last_spot(const FeeState<T>& fee_state) {
    return fee_state[FEE_STATE_LAST_SPOT];
}

template <typename T>
inline uint64_t fee_state_last_update_ts(const FeeState<T>& fee_state) {
    return static_cast<uint64_t>(fee_state[FEE_STATE_LAST_UPDATE_TS]);
}

template <typename T>
inline void init_fee_state(
    FeeState<T>& fee_state,
    const T& spot,
    uint64_t block_timestamp
) {
    fee_state = zero_fee_state<T>();
    fee_state[FEE_STATE_LAST_SPOT] = spot;
    fee_state[FEE_STATE_LAST_UPDATE_TS] = static_cast<T>(block_timestamp);
}

template <typename T>
inline void step_fee_state(
    const FeeParams<T>& fee_params,
    FeeState<T>& fee_state,
    const FeeStateInputs<T>& in
) {
    const T fee_volatility_ref = fee_params[FEE_PARAM_FEE_VOLATILITY_REF];
    const uint64_t prev_ts = fee_state_last_update_ts(fee_state);
    const T prev_spot = fee_state_last_spot(fee_state);

    if (
        in.block_timestamp > prev_ts &&
        in.ma_time > T(0) &&
        prev_spot > T(0) &&
        in.spot > T(0) &&
        fee_volatility_ref >= T(0)
    ) {
        const T dt = static_cast<T>(in.block_timestamp - prev_ts);
        const T ret_abs = relative_gap(in.spot, prev_spot, in.precision);
        T alpha = T(0);
        if constexpr (std::is_same_v<T, uint256>) {
            using int256 = boost::multiprecision::number<
                boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
            auto neg = int256(
                -(
                    int256(dt) *
                    int256(in.precision) /
                    int256(in.ma_time)
                )
            );
            alpha = MathOps<T>::wad_exp(neg);
            fee_state[FEE_STATE_VOL_EMA] = (
                ret_abs * (in.precision - alpha) + fee_state[FEE_STATE_VOL_EMA] * alpha
            ) / in.precision;
        } else {
            alpha = T(std::exp(-static_cast<double>(dt) / static_cast<double>(in.ma_time)));
            fee_state[FEE_STATE_VOL_EMA] = ret_abs * (T(1) - alpha) + fee_state[FEE_STATE_VOL_EMA] * alpha;
        }
    }

    fee_state[FEE_STATE_LAST_SPOT] = in.spot;
    fee_state[FEE_STATE_LAST_UPDATE_TS] = static_cast<T>(in.block_timestamp);
}

template <typename T>
inline T calm_discount_from_params(
    const FeeParams<T>& fee_params,
    const FeeState<T>& fee_state,
    const T& inventory_fee,
    const T& precision
) {
    const T calm_discount_max = fee_params[FEE_PARAM_CALM_DISCOUNT_MAX];
    const T fee_volatility_ref = fee_params[FEE_PARAM_FEE_VOLATILITY_REF];
    if (!(calm_discount_max > T(0)) || !(fee_volatility_ref > T(0))) {
        return T(0);
    }

    const T stress_score = scaled_stress_score(
        fee_state_vol_ema(fee_state),
        fee_volatility_ref,
        precision
    );
    const T calm_score = precision - stress_score;
    return (inventory_fee * calm_discount_max * calm_score) / precision / precision;
}

template <typename T>
inline T directional_gap_from_inputs(
    const FeeInputs<T>& in,
    const T& precision
) {
    if (!in.has_trade_context) {
        return T(0);
    }
    if (!(in.spot > T(0)) || !(in.cex_price > T(0))) {
        return T(0);
    }

    if (in.idx_i == 0 && in.idx_j == 1) {
        if (in.cex_price > in.spot) {
            return (in.cex_price * precision) / in.spot - precision;
        }
        return T(0);
    }
    if (in.idx_i == 1 && in.idx_j == 0) {
        if (in.spot > in.cex_price) {
            return (in.spot * precision) / in.cex_price - precision;
        }
        return T(0);
    }
    return T(0);
}

template <typename T>
inline FeeBreakdown<T> eval_fee_model(
    const FeeParams<T>& fee_params,
    const FeeState<T>& fee_state,
    const FeeInputs<T>& in
) {
    FeeBreakdown<T> out;

    const T precision = in.precision;
    const T fee_precision = in.fee_precision;

    const T base_floor = fee_params[FEE_PARAM_BASE_FLOOR];
    const T inventory_fee = inventory_fee_from_params(in.xp, fee_params, precision);
    T calm_discount = calm_discount_from_params(
        fee_params,
        fee_state,
        inventory_fee,
        precision
    );
    if (calm_discount > inventory_fee) {
        calm_discount = inventory_fee;
    }

    const T inventory_component = inventory_fee - calm_discount;
    const T base_component = fee_max(base_floor, inventory_component);

    const T directional_gap = directional_gap_from_inputs(in, precision);
    const T gap_discount = fee_params[FEE_PARAM_GAP_FEE_CONST_DISCOUNT];
    const T gap_scale = fee_params[FEE_PARAM_GAP_FEE_SCALE];
    T gap_addon = T(0);
    if (gap_scale > T(0) && directional_gap > gap_discount) {
        const T excess_gap = directional_gap - gap_discount;
        gap_addon = (fee_precision * gap_scale * excess_gap) / precision / precision;
    }

    out.components[FEE_COMPONENT_BASE] = base_component;
    out.components[FEE_COMPONENT_INVENTORY] = inventory_component;
    out.components[FEE_COMPONENT_GAP] = gap_addon;
    out.signals[FEE_SIGNAL_BALANCE] = balance_indicator(in.xp);
    out.signals[FEE_SIGNAL_VOLATILITY] = scaled_stress_score(
        fee_state_vol_ema(fee_state),
        fee_params[FEE_PARAM_FEE_VOLATILITY_REF],
        precision
    );
    out.signals[FEE_SIGNAL_DIRECTIONAL_GAP] = directional_gap;

    out.total_fee = base_component + gap_addon;
    if (out.total_fee > fee_precision) {
        out.total_fee = fee_precision;
    }
    return out;
}

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
