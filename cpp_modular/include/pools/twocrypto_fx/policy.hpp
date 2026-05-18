// Value-type external policy models for the twocrypto simulator.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "stableswap_math.hpp"

namespace arb {
namespace pools {
namespace twocrypto_fx {

enum class PolicyKind {
    None,
    TwocryptoPolicy,
    ZeroStub,
    OracleX2SequentialFee,
    FixedFee,
};

inline PolicyKind policy_kind_from_string(const std::string& kind) {
    if (kind.empty() || kind == "none") {
        return PolicyKind::None;
    }
    if (kind == "twocrypto_policy") {
        return PolicyKind::TwocryptoPolicy;
    }
    if (kind == "zero_stub") {
        return PolicyKind::ZeroStub;
    }
    if (
        kind == "oracle_x2" ||
        kind == "price_oracle_x2" ||
        kind == "oracle_x2_sequential_fee"
    ) {
        return PolicyKind::OracleX2SequentialFee;
    }
    if (kind == "fixed_fee" || kind == "fixed_fee_policy") {
        return PolicyKind::FixedFee;
    }
    throw std::invalid_argument("unsupported policy kind: " + kind);
}

template <typename T>
struct PolicyConfig {
    PolicyKind kind{PolicyKind::None};
    T fee{T(0)};
};

template <typename T>
struct PolicyState {
    std::array<T, 2> xp{T(0), T(0)};
    T price_scale{T(0)};
    T price_oracle{T(0)};
    T last_prices{T(0)};
    T virtual_price{T(0)};
    T xcp_profit{T(0)};
    T D{T(0)};
    T policy_fee{T(0)};
    uint64_t ts{0};
    uint64_t update_nonce{0};
};

template <typename T>
struct PolicyPoolConfig {
    T mid_fee{T(0)};
    T out_fee{T(0)};
    T fee_gamma{T(0)};
    T ma_time{T(1)};
    T precision{T(1)};
    T fee_precision{T(1)};
};

template <typename T>
struct PolicyCallContext {
    uint64_t block_timestamp{0};
    T price_oracle{T(0)};
};

template <typename T>
class PolicyModel {
public:
    PolicyKind kind = PolicyKind::None;
    PolicyConfig<T> params{};
    PolicyState<T> state{};
    PolicyPoolConfig<T> config{};
    PolicyCallContext<T> call_context{};

    PolicyModel() = default;
    explicit PolicyModel(PolicyKind _kind) : kind(_kind) {
        params.kind = _kind;
    }
    explicit PolicyModel(const PolicyConfig<T>& _params) : kind(_params.kind), params(_params) {}

    void configure_pool(
        const T& mid_fee,
        const T& out_fee,
        const T& fee_gamma,
        const T& ma_time,
        const T& precision,
        const T& fee_precision
    ) {
        config.mid_fee = mid_fee;
        config.out_fee = out_fee;
        config.fee_gamma = fee_gamma;
        config.ma_time = ma_time;
        config.precision = precision;
        config.fee_precision = fee_precision;
    }

    void prepare_price_scale_call(uint64_t block_timestamp, const T& price_oracle) {
        call_context.block_timestamp = block_timestamp;
        call_context.price_oracle = price_oracle;
    }

    T get_fee(const std::array<T, 2>& xp) const {
        if (kind == PolicyKind::FixedFee) {
            return fixed_fee(xp);
        }
        if (kind == PolicyKind::OracleX2SequentialFee) {
            return oracle_x2_sequential_fee(xp);
        }
        if (kind == PolicyKind::TwocryptoPolicy) {
            return twocrypto_policy_fee(xp);
        }
        return zero_stub_fee(xp);
    }

    T get_price_scale() const {
        if (kind == PolicyKind::None || kind == PolicyKind::ZeroStub || kind == PolicyKind::FixedFee) {
            return T(0);
        }
        if (kind == PolicyKind::OracleX2SequentialFee) {
            return call_context.price_oracle + call_context.price_oracle;
        }

        const auto& s = state;
        if (s.ts == 0 || s.price_scale == T(0)) {
            return T(0);
        }

        T price_scale = s.price_scale;
        T price_oracle = s.price_oracle;
        if (s.ts < call_context.block_timestamp) {
            T dt = T(call_context.block_timestamp - s.ts);
            if constexpr (std::is_same_v<T, uint256>) {
                auto neg = int256(
                    -(
                        int256(dt) *
                        int256(config.precision) /
                        int256(config.ma_time)
                    )
                );
                T alpha = MathOps<T>::wad_exp(neg);
                T ema_input = s.last_prices;
                T lower = price_scale / 2;
                if (ema_input < lower) ema_input = lower;
                T upper = price_scale * 2;
                if (ema_input > upper) ema_input = upper;
                price_oracle = (
                    ema_input * (config.precision - alpha) + price_oracle * alpha
                ) / config.precision;
            } else {
                auto alpha = std::exp(
                    -static_cast<double>(dt) / static_cast<double>(config.ma_time)
                );
                T ema_input = s.last_prices;
                T lower = price_scale / 2;
                if (ema_input < lower) ema_input = lower;
                T upper = price_scale * 2;
                if (ema_input > upper) ema_input = upper;
                price_oracle = ema_input * (T(1) - T(alpha)) + price_oracle * T(alpha);
            }
        }
        return price_oracle;
    }

    void update_state(
        const std::array<T, 2>& xp,
        const T& price_scale,
        const T& price_oracle,
        const T& last_prices,
        const T& virtual_price,
        const T& xcp_profit,
        const T& d_value,
        uint64_t oracle_timestamp
    ) {
        if (kind == PolicyKind::None || kind == PolicyKind::ZeroStub || kind == PolicyKind::FixedFee) {
            return;
        }

        state.xp = xp;
        state.price_scale = price_scale;
        state.price_oracle = price_oracle;
        state.last_prices = last_prices;
        state.virtual_price = virtual_price;
        state.xcp_profit = xcp_profit;
        state.D = d_value;
        state.ts = oracle_timestamp;

        if (kind == PolicyKind::OracleX2SequentialFee) {
            state.update_nonce += 1;
            state.policy_fee = sequence_fee(
                xp,
                price_scale,
                price_oracle,
                last_prices,
                virtual_price,
                xcp_profit,
                d_value,
                oracle_timestamp,
                state.update_nonce,
                config.fee_precision
            );
        }
    }

private:
    static constexpr uint64_t MIN_SEQUENCE_FEE_BPS = 1ULL;
    static constexpr uint64_t MAX_SEQUENCE_FEE_BPS = 1000ULL;
    static constexpr uint64_t SEQUENCE_CYCLE_LENGTH = 100ULL;
    static constexpr uint64_t SEQUENCE_HALF_CYCLE = 50ULL;
    static constexpr uint64_t BPS_SCALE = 10000ULL;

    static uint64_t sequence_fee_bps(uint64_t update_nonce) {
        if (update_nonce == 0) {
            return 0;
        }
        uint64_t idx = (update_nonce - 1) % SEQUENCE_CYCLE_LENGTH;
        uint64_t leg = (idx < SEQUENCE_HALF_CYCLE)
            ? idx
            : (SEQUENCE_CYCLE_LENGTH - 1 - idx);
        return MIN_SEQUENCE_FEE_BPS + (
            leg * (MAX_SEQUENCE_FEE_BPS - MIN_SEQUENCE_FEE_BPS)
        ) / (SEQUENCE_HALF_CYCLE - 1);
    }

    static T sequence_fee(
        const std::array<T, 2>& xp,
        const T& price_scale,
        const T& price_oracle,
        const T& last_prices,
        const T& virtual_price,
        const T& xcp_profit,
        const T& d_value,
        uint64_t oracle_timestamp,
        uint64_t update_nonce,
        const T& fee_precision
    ) {
        (void)xp;
        (void)price_scale;
        (void)price_oracle;
        (void)last_prices;
        (void)virtual_price;
        (void)xcp_profit;
        (void)d_value;
        (void)oracle_timestamp;

        uint64_t fee_bps = sequence_fee_bps(update_nonce);
        return T(fee_bps) * fee_precision / T(BPS_SCALE);
    }

    T fixed_fee(const std::array<T, 2>& xp) const {
        (void)xp;
        return params.fee;
    }

    T zero_stub_fee(const std::array<T, 2>& xp) const {
        (void)xp;
        return T(0);
    }

    T oracle_x2_sequential_fee(const std::array<T, 2>& xp) const {
        (void)xp;
        return state.policy_fee;
    }

    T twocrypto_policy_fee(const std::array<T, 2>& xp) const {
        if (config.fee_gamma == T(0)) {
            return config.mid_fee;
        }

        T bsum = xp[0] + xp[1];
        if (bsum == T(0)) {
            return config.mid_fee;
        }

        T balance_term = config.precision * T(4) * xp[0] / bsum * xp[1] / bsum;
        balance_term = config.fee_gamma * balance_term /
            (config.fee_gamma * balance_term / config.precision + config.precision - balance_term);

        return (
            config.mid_fee * balance_term +
            config.out_fee * (config.precision - balance_term)
        ) / config.precision;
    }
};

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
