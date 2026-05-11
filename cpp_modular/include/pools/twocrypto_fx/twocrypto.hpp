// TwoCrypto pool (templated on numeric type)
// Duplicated from cpp/include/twocrypto.hpp with namespace arb::pools::twocrypto_fx
#pragma once

#include <string>
#include <array>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "stableswap_math.hpp"

namespace arb {
namespace pools {
namespace twocrypto_fx {

// Pool traits for numeric types
template <typename T>
struct PoolTraits;

template <>
struct PoolTraits<uint256> {
    using T = uint256;
    static T PRECISION() { static T v("1000000000000000000"); return v; }
    static T FEE_PRECISION() { static T v("10000000000"); return v; }
    static T A_MULTIPLIER() { return T(10000); }
    static T NOISE_FEE() { static T v("100000"); return v; }
    static T ZERO() { return T(0); }
    static T ONE() { return T(1); }
    static T ROUNDING_UNIT_XP() { return T(1); }
    static bool is_zero(const T& x) { return x == 0; }
    static T max(const T& a, const T& b) { return a > b ? a : b; }
    static T min(const T& a, const T& b) { return a < b ? a : b; }
};

template <>
struct PoolTraits<double> {
    using T = double;
    static T PRECISION() { return 1.0; }
    static T FEE_PRECISION() { return 1.0; }
    static T A_MULTIPLIER() { return 10000.0; }
    static T NOISE_FEE() { return 1e-5; }
    static T ZERO() { return 0.0; }
    static T ONE() { return 1.0; }
    static T ROUNDING_UNIT_XP() { return 0.0; }
    static bool is_zero(const T& x) { return std::abs(x) <= 0.0; }
    static T max(const T& a, const T& b) { return a > b ? a : b; }
    static T min(const T& a, const T& b) { return a < b ? a : b; }
};

template <>
struct PoolTraits<float> {
    using T = float;
    static T PRECISION() { return 1.0f; }
    static T FEE_PRECISION() { return 1.0f; }
    static T A_MULTIPLIER() { return 10000.0f; }
    static T NOISE_FEE() { return 1e-5f; }
    static T ZERO() { return 0.0f; }
    static T ONE() { return 1.0f; }
    static T ROUNDING_UNIT_XP() { return 0.0f; }
    static bool is_zero(const T& x) { return std::abs(x) <= 0.0f; }
    static T max(const T& a, const T& b) { return a > b ? a : b; }
    static T min(const T& a, const T& b) { return a < b ? a : b; }
};

template <>
struct PoolTraits<long double> {
    using T = long double;
    static T PRECISION() { return 1.0L; }
    static T FEE_PRECISION() { return 1.0L; }
    static T A_MULTIPLIER() { return 10000.0L; }
    static T NOISE_FEE() { return 1e-5L; }
    static T ZERO() { return 0.0L; }
    static T ONE() { return 1.0L; }
    static T ROUNDING_UNIT_XP() { return 0.0L; }
    static bool is_zero(const T& x) { return fabsl(x) <= 0.0L; }
    static T max(const T& a, const T& b) { return a > b ? a : b; }
    static T min(const T& a, const T& b) { return a < b ? a : b; }
};

enum class PolicyKind {
    None,
    TwocryptoPolicy,
    ZeroStub,
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
    throw std::invalid_argument("unsupported policy kind: " + kind);
}

template <typename T>
struct PolicyState {
    std::array<T, 2> xp{PoolTraits<T>::ZERO(), PoolTraits<T>::ZERO()};
    T price_scale{PoolTraits<T>::ZERO()};
    T price_oracle{PoolTraits<T>::ZERO()};
    T last_prices{PoolTraits<T>::ZERO()};
    T virtual_price{PoolTraits<T>::ZERO()};
    T xcp_profit{PoolTraits<T>::ZERO()};
    T D{PoolTraits<T>::ZERO()};
    uint64_t ts{0};
};

template <typename T>
class TwoCryptoPool {
public:
    using Ops = MathOps<T>;
    using Traits = PoolTraits<T>;
    static constexpr int N_COINS = 2;

    // State variables
    std::array<T, 2> balances{Traits::ZERO(), Traits::ZERO()};
    T D = Traits::ZERO();
    T totalSupply = Traits::ZERO();

    // Price variables
    T cached_price_scale = Traits::PRECISION();
    T cached_price_oracle = Traits::PRECISION();
    T last_prices = Traits::PRECISION();
    uint64_t last_timestamp = 0;

    // Parameters (normalized)
    T A = Traits::ZERO();
    T gamma = Traits::ZERO();
    T mid_fee = Traits::ZERO();
    T out_fee = Traits::ZERO();
    T fee_gamma = Traits::ZERO();
    T adjustment_step_min = Traits::ZERO();
    T adjustment_step_max = Traits::ZERO();
    T ma_time = Traits::ONE();
    T reserved_profit_fraction = Traits::FEE_PRECISION() / 2;
    T admin_fee = Traits::FEE_PRECISION() / 2;
    PolicyKind policy_kind = PolicyKind::None;
    PolicyState<T> policy_state{};

    // Profit tracking
    T xcp_profit = Traits::PRECISION();
    T lp_xcp_profit = Traits::PRECISION();
    T xcp_profit_a = Traits::PRECISION();
    T virtual_price = Traits::PRECISION();
    std::array<T, 2> admin_balances{Traits::ZERO(), Traits::ZERO()};

    // Token precisions
    std::array<T, 2> precisions{Traits::ONE(), Traits::ONE()};

    // Time for testing
    uint64_t block_timestamp = 0;

    // Donations
    T donation_shares = Traits::ZERO();
    T donation_shares_max_ratio = Traits::PRECISION() * 10 / 100; // 10%
    T donation_duration = T(7 * 86400);
    // T donation_duration = T(600);
    T last_donation_release_ts = Traits::ZERO();
    T donation_protection_expiry_ts = Traits::ZERO();
    T donation_protection_period = T(10 * 60);
    T donation_protection_lp_threshold = Traits::PRECISION() * 20 / 100; // 20%
    T donation_protection_extension_remainder = Traits::ZERO();

    uint64_t last_admin_fee_claim_timestamp = 0;

public:
    TwoCryptoPool(
        const std::array<T, 2>& _precisions,
        const T& _A,
        const T& _gamma,
        const T& _mid_fee,
        const T& _out_fee,
        const T& _fee_gamma,
        const T& _adjustment_step_min,
        const T& _adjustment_step_max,
        const T& _ma_time,
        const T& initial_price,
        const T& _reserved_profit_fraction = PoolTraits<T>::FEE_PRECISION() / 2,
        const T& _admin_fee = PoolTraits<T>::FEE_PRECISION() / 2,
        PolicyKind _policy_kind = PolicyKind::None
    ) {
        precisions = _precisions;
        A = _A; gamma = _gamma;
        mid_fee = _mid_fee; out_fee = _out_fee; fee_gamma = _fee_gamma;
        adjustment_step_min = _adjustment_step_min;
        adjustment_step_max = _adjustment_step_max;
        ma_time = _ma_time;
        reserved_profit_fraction = _reserved_profit_fraction;
        admin_fee = _admin_fee;
        policy_kind = _policy_kind;

        cached_price_scale = initial_price;
        cached_price_oracle = initial_price;
        last_prices = initial_price;

        xcp_profit = Traits::PRECISION();
        lp_xcp_profit = Traits::PRECISION();
        xcp_profit_a = Traits::PRECISION();
        virtual_price = Traits::PRECISION();

        block_timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        last_timestamp = block_timestamp;
    }

private:
    T _clamp_fee(const T& fee) const {
        const T min_fee = Traits::NOISE_FEE();
        const T max_fee = Traits::FEE_PRECISION();
        return Traits::min(max_fee, Traits::max(min_fee, fee));
    }

    T _native_fee(const std::array<T, 2>& xp) const {
        if (fee_gamma == Traits::ZERO()) {
            return mid_fee;
        }

        T Bsum = xp[0] + xp[1];
        if (Bsum == Traits::ZERO()) {
            return mid_fee;
        }

        T B = Traits::PRECISION() * N_COINS * N_COINS * xp[0] / Bsum * xp[1] / Bsum;
        B = fee_gamma * B /
            (fee_gamma * B / Traits::PRECISION() + Traits::PRECISION() - B);

        return (
            mid_fee * B + out_fee * (Traits::PRECISION() - B)
        ) / Traits::PRECISION();
    }

    // _xp: balances scaled to common precision with price_scale for coin1
    std::array<T, 2> _xp(
        const std::array<T, 2>& _balances,
        const T& price_scale
    ) const {
        return {
            _balances[0] * precisions[0],
            _balances[1] * precisions[1] * price_scale / Traits::PRECISION()
        };
    }

    // _fee: dynamic fee between mid_fee and out_fee based on balance skew
    T _fee(const std::array<T, 2>& xp) const {
        if (policy_kind == PolicyKind::ZeroStub) {
            return _clamp_fee(_native_fee(xp));
        }
        if (policy_kind == PolicyKind::TwocryptoPolicy) {
            T policy_fee = _native_fee(xp);
            if (policy_fee != Traits::ZERO()) {
                return _clamp_fee(policy_fee);
            }
        }
        return _clamp_fee(_native_fee(xp));
    }

    T _policy_price_target(uint64_t ts) const {
        if (policy_kind == PolicyKind::None || policy_kind == PolicyKind::ZeroStub) {
            return Traits::ZERO();
        }
        const auto& state = policy_state;
        if (state.ts == 0) {
            return Traits::ZERO();
        }

        T price_scale = state.price_scale;
        T price_oracle = state.price_oracle;
        if (price_scale == Traits::ZERO()) {
            return Traits::ZERO();
        }

        if (state.ts < ts) {
            T dt = T(ts - state.ts);
            if constexpr (std::is_same_v<T, uint256>) {
                auto neg = int256(
                    -(
                        int256(dt) *
                        int256(Traits::PRECISION()) /
                        int256(ma_time)
                    )
                );
                T alpha = MathOps<T>::wad_exp(neg);
                T ema_input = state.last_prices;
                T lower = price_scale / 2;
                if (ema_input < lower) ema_input = lower;
                T upper = price_scale * 2;
                if (ema_input > upper) ema_input = upper;
                price_oracle = (
                    ema_input * (Traits::PRECISION() - alpha) + price_oracle * alpha
                ) / Traits::PRECISION();
            } else {
                auto alpha = std::exp(
                    - static_cast<double>(dt) / static_cast<double>(ma_time)
                );
                T ema_input = state.last_prices;
                T lower = price_scale / 2;
                if (ema_input < lower) ema_input = lower;
                T upper = price_scale * 2;
                if (ema_input > upper) ema_input = upper;
                price_oracle = ema_input * (T(1) - T(alpha)) + price_oracle * T(alpha);
            }
        }
        return price_oracle;
    }

    void _update_policy_state(
        const std::array<T, 2>& xp,
        const T& price_scale,
        const T& price_oracle,
        const T& prices,
        const T& vp,
        const T& profit,
        const T& d_value
    ) {
        policy_state.xp = xp;
        policy_state.price_scale = price_scale;
        policy_state.price_oracle = price_oracle;
        policy_state.last_prices = prices;
        policy_state.virtual_price = vp;
        policy_state.xcp_profit = profit;
        policy_state.D = d_value;
        policy_state.ts = last_timestamp;
    }

    T _apply_admin_d_token_fee(
        std::array<T, 2>& local_balances,
        const T& d_token_fee,
        const T& fee_supply
    ) {
        T admin_d_token_fee = (
            d_token_fee * reserved_profit_fraction * admin_fee
        ) / (Traits::FEE_PRECISION() * Traits::FEE_PRECISION());
        if (admin_d_token_fee > Traits::ZERO() && fee_supply > Traits::ZERO()) {
            for (size_t i = 0; i < N_COINS; ++i) {
                T admin_amount = local_balances[i] * admin_d_token_fee / fee_supply;
                admin_balances[i] += admin_amount;
                local_balances[i] -= admin_amount;
            }
        }
        return admin_d_token_fee;
    }

    // _xcp: cross-product invariant in xcp units
    T _xcp(const T& _D, const T& price_scale) const {
        if constexpr (std::is_same_v<T, uint256>) {
            auto sqrt_price = boost::multiprecision::sqrt(
                Traits::PRECISION() * price_scale
            );
            return _D * Traits::PRECISION() / N_COINS / sqrt_price;
        } else if constexpr (std::is_same_v<T, long double>) {
            long double sp = std::sqrt(static_cast<long double>(Traits::PRECISION() * price_scale));
            return _D * Traits::PRECISION() / T(N_COINS) / T(sp);
        } else {
            double sp = std::sqrt(static_cast<double>(Traits::PRECISION() * price_scale));
            return _D * Traits::PRECISION() / T(N_COINS) / T(sp);
        }
    }

    // _donation_shares: unlocked donation supply (optionally protected)
    T _donation_shares(bool donation_protection = true) const {
        if (donation_shares == Traits::ZERO()) {
            return Traits::ZERO();
        }

        T elapsed  = T(block_timestamp) - last_donation_release_ts;
        T unlocked = donation_shares * elapsed / donation_duration;
        if (unlocked > donation_shares) {
            unlocked = donation_shares;
        }

        if (!donation_protection) {
            return unlocked;
        }

        T protection_factor = Traits::ZERO();
        if (donation_protection_expiry_ts > T(block_timestamp)) {
            protection_factor = (
                (donation_protection_expiry_ts - T(block_timestamp)) *
                Traits::PRECISION() /
                donation_protection_period
            );
            if (protection_factor > Traits::PRECISION()) {
                protection_factor = Traits::PRECISION();
            }
        }

        return (
            unlocked * (Traits::PRECISION() - protection_factor)
        ) / Traits::PRECISION();
    }

    // _calc_token_fee: liquidity op fee approximation
    T _calc_token_fee(
        const std::array<T, 2>& amounts,
        const std::array<T, 2>& xp,
        bool donation,
        bool deposit
    ) const {
        if (donation) {
            return Traits::NOISE_FEE();
        }

        T denom = balances[1] * precisions[1];
        T balances_ratio = Traits::ZERO();
        if (denom > Traits::ZERO()) {
            balances_ratio = (
                balances[0] * precisions[0] * Traits::PRECISION()
            ) / denom;
        }

        std::array<T, 2> amounts_scaled = {
            amounts[0] * precisions[0],
            amounts[1] * precisions[1] * balances_ratio / Traits::PRECISION()
        };

        T fee_prime = _fee(xp) * N_COINS / (4 * (N_COINS - 1));

        T S = amounts_scaled[0] + amounts_scaled[1];
        if (S == Traits::ZERO()) {
            return Traits::NOISE_FEE();
        }

        T avg   = S / N_COINS;
        T diff0 = (amounts_scaled[0] > avg) ? (amounts_scaled[0] - avg)
                                            : (avg - amounts_scaled[0]);
        T diff1 = (amounts_scaled[1] > avg) ? (amounts_scaled[1] - avg)
                                            : (avg - amounts_scaled[1]);
        T Sdiff = diff0 + diff1;

        T lp_spam_penalty_fee = Traits::ZERO();
        if (deposit && donation_protection_expiry_ts > T(block_timestamp)) {
            T protection_factor = (
                (donation_protection_expiry_ts - T(block_timestamp)) *
                Traits::PRECISION() /
                donation_protection_period
            );
            if (protection_factor > Traits::PRECISION()) {
                protection_factor = Traits::PRECISION();
            }
            T donation_ratio = (totalSupply > Traits::ZERO())
                ? donation_shares * Traits::PRECISION() / totalSupply
                : Traits::ZERO();
            lp_spam_penalty_fee = (
                protection_factor * fee_prime * donation_ratio / Traits::PRECISION()
            ) / donation_shares_max_ratio;
            if (lp_spam_penalty_fee > fee_prime) {
                lp_spam_penalty_fee = fee_prime;
            }
        }

        return fee_prime * Sdiff / S + Traits::NOISE_FEE() + lp_spam_penalty_fee;
    }

public:
    // Cheap tick to update EMA/oracle and possibly adjust price_scale without a swap
    void tick() {
        auto A_gamma = std::array<T, 2>{ A, gamma };
        const auto xp = _xp(balances, cached_price_scale);
        cached_price_scale = tweak_price(A_gamma, xp, D, virtual_price);
    }

    // add_liquidity: deposit into the pool; supports donation mode with cap semantics
    T add_liquidity(
        const std::array<T, 2>& amounts,
        T min_mint_amount,
        bool donation = false
    ) {
        if (amounts[0] + amounts[1] == Traits::ZERO()) {
            throw std::invalid_argument("no coins to add");
        }

        auto old_balances = balances;
        auto new_balances = std::array<T, 2>{
            balances[0] + amounts[0],
            balances[1] + amounts[1]
        };

        T price_scale = cached_price_scale;
        auto xp     = _xp(new_balances, price_scale);
        auto old_xp = _xp(old_balances, price_scale);
        (void)old_xp;

        auto A_gamma = std::array<T, 2>{ A, gamma };

        T old_D = D;
        T D_new = Ops::newton_D(A_gamma[0], A_gamma[1], xp, T(0));
        T token_supply = totalSupply;
        T vp_preop = virtual_price;
        if (old_D > Traits::ZERO() && token_supply > Traits::ZERO()) {
            vp_preop = Traits::PRECISION() * _xcp(old_D, price_scale) / token_supply;
        }

        T d_token = Traits::ZERO();
        if (old_D > Traits::ZERO()) {
            d_token = token_supply * D_new / old_D - token_supply;
        } else {
            d_token = _xcp(D_new, price_scale);
        }

        if (old_D > Traits::ZERO()) {
            T approx_fee  = _calc_token_fee(amounts, xp, donation, /*deposit=*/true);
            T d_token_fee = approx_fee * d_token / PoolTraits<T>::FEE_PRECISION();
            if constexpr (std::is_same_v<T, uint256>) {
                d_token_fee += Traits::ONE();
            }
            d_token -= d_token_fee;
            if (!donation && d_token_fee > Traits::ZERO() &&
                reserved_profit_fraction > Traits::ZERO() && admin_fee > Traits::ZERO()) {
                T fee_supply = token_supply + d_token + d_token_fee;
                _apply_admin_d_token_fee(new_balances, d_token_fee, fee_supply);
                xp = _xp(new_balances, price_scale);
                D_new = Ops::newton_D(A_gamma[0], A_gamma[1], xp, T(0));
            }
        }

        // Constraints before commit
        if (old_D > Traits::ZERO() && donation) {
            T new_donation_shares = donation_shares + d_token;
            T ratio = (
                new_donation_shares * Traits::PRECISION()
            ) / (token_supply + d_token);
            if (ratio > donation_shares_max_ratio) {
                throw std::runtime_error("donation above cap");
            }
        }
        if (d_token < min_mint_amount) {
            throw std::runtime_error("slippage");
        }

        // Commit
        balances = new_balances;
        if (old_D > Traits::ZERO()) {
            D = D_new;

            if (donation) {
                T new_donation_shares = donation_shares + d_token;
                T unlocked = _donation_shares(false);
                T new_elapsed = Traits::ZERO();
                if (new_donation_shares > Traits::ZERO()) {
                    new_elapsed = (unlocked * donation_duration) / new_donation_shares;
                }
                last_donation_release_ts = T(block_timestamp) - new_elapsed;
                donation_shares = new_donation_shares;
                totalSupply += d_token;
            } else {
                T relative_lp_add = (
                    d_token * Traits::PRECISION()
                ) / (token_supply + d_token);

                if (relative_lp_add > Traits::ZERO() && donation_shares > Traits::ZERO()) {
                    T raw_extension = (
                        relative_lp_add * donation_protection_period
                    ) + donation_protection_extension_remainder;
                    T extension_seconds = raw_extension / donation_protection_lp_threshold;

                    T current_expiry = (
                        donation_protection_expiry_ts > T(block_timestamp)
                    ) ? donation_protection_expiry_ts : T(block_timestamp);

                    T uncapped_expiry = current_expiry + extension_seconds;
                    T max_expiry = T(block_timestamp) + donation_protection_period;

                    if (uncapped_expiry >= max_expiry) {
                        donation_protection_expiry_ts = max_expiry;
                        donation_protection_extension_remainder = Traits::ZERO();
                    } else {
                        donation_protection_expiry_ts = uncapped_expiry;
                        if constexpr (std::is_same_v<T, uint256>) {
                            donation_protection_extension_remainder = raw_extension % donation_protection_lp_threshold;
                        } else {
                            donation_protection_extension_remainder = std::fmod(
                                raw_extension,
                                donation_protection_lp_threshold
                            );
                        }
                    }
                }
                totalSupply += d_token;
            }

            cached_price_scale = tweak_price(A_gamma, xp, D_new, vp_preop);
        } else {
            D = D_new;
            virtual_price = Traits::PRECISION();
            xcp_profit    = Traits::PRECISION();
            lp_xcp_profit = Traits::PRECISION();
            xcp_profit_a  = Traits::PRECISION();
            totalSupply  += d_token;
            _update_policy_state(xp, price_scale, cached_price_oracle, last_prices, virtual_price, xcp_profit, D);
        }
        return d_token;
    }

    // remove_liquidity: burn LP to withdraw proportionally
    std::array<T, 2> remove_liquidity(
        T amount,
        const std::array<T, 2>& min_amounts
    ) {
        if (amount > totalSupply) {
            throw std::invalid_argument("insufficient LP tokens");
        }

        std::array<T, 2> withdrawn{};
        for (size_t i = 0; i < N_COINS; ++i) {
            withdrawn[i] = balances[i] * amount / totalSupply;
            if (withdrawn[i] < min_amounts[i]) {
                throw std::runtime_error("withdrawal resulted in fewer coins than expected");
            }
            balances[i] -= withdrawn[i];
        }

        T old_total_supply = totalSupply;
        totalSupply -= amount;

        if (old_total_supply > Traits::ZERO()) {
            D = D - (D * amount / old_total_supply);
        } else {
            D = Traits::ZERO();
        }

        return withdrawn;
    }

    // exchange: swap coin i for coin j
    std::array<T, 3> exchange(
        T i,
        T j,
        T dx,
        T min_dy
    ) {
        size_t idx_i = static_cast<size_t>(i);
        size_t idx_j = static_cast<size_t>(j);

        if (idx_i == idx_j || idx_i >= N_COINS || idx_j >= N_COINS) {
            throw std::invalid_argument("coin index out of range");
        }
        if (dx == Traits::ZERO()) {
            throw std::invalid_argument("zero dx");
        }

        T price_scale = cached_price_scale;
        T vp_preop = (totalSupply > Traits::ZERO())
            ? (Traits::PRECISION() * _xcp(D, price_scale) / totalSupply)
            : virtual_price;

        auto balances_local = balances;
        balances_local[idx_i] += dx;
        auto xp = _xp(balances_local, price_scale);

        auto A_gamma = std::array<T, 2>{ A, gamma };

        auto y_out = Ops::get_y(A_gamma[0], A_gamma[1], xp, D, idx_j);
        T dy_xp = xp[idx_j] - y_out.value;
        xp[idx_j] -= dy_xp;

        T dy_tokens = dy_xp - PoolTraits<T>::ROUNDING_UNIT_XP();
        if (idx_j > 0) {
            dy_tokens = dy_tokens * Traits::PRECISION() / price_scale;
        }
        dy_tokens = dy_tokens / precisions[idx_j];

        T fee = _fee(xp) * dy_tokens / PoolTraits<T>::FEE_PRECISION();
        T dy_after_fee = dy_tokens - fee;
        if (dy_after_fee < min_dy) {
            throw std::runtime_error("slippage");
        }

        balances[idx_i] += dx;
        balances[idx_j] -= dy_after_fee;
        auto xp_new = _xp(balances, price_scale);
        T D_new = Ops::newton_D(A_gamma[0], A_gamma[1], xp_new, 0);
        D = D_new;
        T admin_fee_amount = (
            fee * reserved_profit_fraction * admin_fee
        ) / (Traits::FEE_PRECISION() * Traits::FEE_PRECISION());
        if (admin_fee_amount > Traits::ZERO()) {
            admin_balances[idx_j] += admin_fee_amount;
            balances[idx_j] -= admin_fee_amount;
            xp_new = _xp(balances, price_scale);
            D_new = Ops::newton_D(A_gamma[0], A_gamma[1], xp_new, 0);
            D = D_new;
        }
        T new_price_scale = tweak_price(A_gamma, xp_new, D_new, vp_preop);
        return { dy_after_fee, fee, new_price_scale };
    }

    T tweak_price(
        const std::array<T, 2>& _A_gamma,
        const std::array<T, 2>& xp,
        T _D,
        T vp_preop
    ) {
        static const bool trace = []() {
            if (const char* env = std::getenv("TRACE")) {
                return std::string(env) == "1";
            }
            return false;
        }();
        T price_oracle = cached_price_oracle;
        T price_scale  = cached_price_scale;

        // EMA update
        uint64_t last_ts = last_timestamp;
        if (last_ts < block_timestamp) {
            T dt = T(block_timestamp - last_ts);
            if constexpr (std::is_same_v<T, uint256>) {
                auto neg = int256(
                    -(
                        int256(dt) *
                        int256(PoolTraits<T>::PRECISION()) /
                        int256(ma_time)
                    )
                );
                T alpha  = MathOps<T>::wad_exp(neg);
                T capped = last_prices;
                T lower = price_scale / 2;
                if (capped < lower) capped = lower;
                if (capped > 2 * price_scale) capped = 2 * price_scale;
                price_oracle = (
                    capped * (PoolTraits<T>::PRECISION() - alpha) + price_oracle * alpha
                ) / PoolTraits<T>::PRECISION();
            } else {
                auto alpha = std::exp(
                    - static_cast<double>(dt) / static_cast<double>(ma_time)
                );
                T capped = last_prices;
                T lower = price_scale / 2;
                if (capped < lower) capped = lower;
                if (capped > 2 * price_scale) capped = 2 * price_scale;

                price_oracle = capped * (T(1) - T(alpha)) + price_oracle * T(alpha);
            }
            cached_price_oracle = price_oracle;
            last_timestamp      = block_timestamp;
        }

        // Update last_prices from current state
        last_prices = (
            MathOps<T>::get_p(xp, _D, _A_gamma) * price_scale
        ) / PoolTraits<T>::PRECISION();

        // Compute current virtual price and profits
        T total_supply      = totalSupply;
        T donation_unlocked = _donation_shares();
        T locked_supply     = total_supply - donation_unlocked;

        T old_virtual_price = virtual_price;
        T xcp               = _xcp(_D, price_scale);
        T vp = (total_supply > PoolTraits<T>::ZERO())
            ? (PoolTraits<T>::PRECISION() * xcp / total_supply)
            : PoolTraits<T>::PRECISION();

        if (vp < vp_preop) {
            throw std::runtime_error("virtual price decreased");
        }

        T old_xcp_profit = xcp_profit;
        if (vp > old_virtual_price) {
            xcp_profit = xcp_profit + vp - old_virtual_price;
            if (xcp_profit > Traits::PRECISION()) {
                T baseline = Traits::max(old_xcp_profit, Traits::PRECISION());
                T d_profit = (xcp_profit > baseline) ? (xcp_profit - baseline) : Traits::ZERO();
                T denom = (
                    Traits::FEE_PRECISION() * Traits::FEE_PRECISION() -
                    reserved_profit_fraction * admin_fee
                );
                if (d_profit > Traits::ZERO() && denom > Traits::ZERO()) {
                    lp_xcp_profit += (
                        d_profit * reserved_profit_fraction * (Traits::FEE_PRECISION() - admin_fee)
                    ) / denom;
                }
            }
        } else {
            T vp_delta = old_virtual_price - vp;
            xcp_profit = (xcp_profit > vp_delta) ? (xcp_profit - vp_delta) : Traits::ZERO();
            if (lp_xcp_profit > Traits::PRECISION() && vp_delta <= (lp_xcp_profit - Traits::PRECISION())) {
                lp_xcp_profit -= vp_delta;
            } else {
                lp_xcp_profit = Traits::PRECISION();
            }
        }

        if (trace) {
            if constexpr (std::is_same_v<T, uint256>) {
                std::cout << "TRACE tp_ema price_oracle=" << price_oracle.template convert_to<std::string>()
                          << " last_prices=" << last_prices.template convert_to<std::string>()
                          << " price_scale=" << price_scale.template convert_to<std::string>()
                          << "\n";
            } else {
                std::cout << "TRACE tp_ema price_oracle=" << price_oracle
                          << " last_prices=" << last_prices
                          << " price_scale=" << price_scale
                          << "\n";
            }
        }

        T vp_boosted = (locked_supply > PoolTraits<T>::ZERO())
            ? (PoolTraits<T>::PRECISION() * xcp / locked_supply)
            : vp;

        if (trace) {
            if constexpr (std::is_same_v<T, uint256>) {
                std::cout << "TRACE tp_gating vp=" << vp.template convert_to<std::string>()
                          << " lp_xcp_profit=" << lp_xcp_profit.template convert_to<std::string>()
                          << " vp_boosted=" << vp_boosted.template convert_to<std::string>()
                          << "\n";
            } else {
                std::cout << "TRACE tp_gating vp=" << vp
                          << " lp_xcp_profit=" << lp_xcp_profit
                          << " vp_boosted=" << vp_boosted
                          << "\n";
            }
        }

        // Price adjustment path
        if ((vp_boosted > lp_xcp_profit) && (last_ts < block_timestamp)) {
            T target_price = price_oracle;
            T p_policy = _policy_price_target(block_timestamp);
            if (p_policy > Traits::ZERO()) {
                T policy_bound = price_oracle / 5;
                T lo = price_oracle - policy_bound;
                T hi = price_oracle + policy_bound;
                target_price = Traits::min(hi, Traits::max(lo, p_policy));
            }

            T norm = target_price * PoolTraits<T>::PRECISION() / price_scale;
            if (norm > PoolTraits<T>::PRECISION()) {
                norm = norm - PoolTraits<T>::PRECISION();
            } else {
                norm = PoolTraits<T>::PRECISION() - norm;
            }

            T step = PoolTraits<T>::min(norm / 5, adjustment_step_max);
            if (trace) {
                if constexpr (std::is_same_v<T, uint256>) {
                    std::cout << "TRACE tp_norm norm=" << norm.template convert_to<std::string>()
                              << " step=" << step.template convert_to<std::string>()
                              << "\n";
                } else {
                    std::cout << "TRACE tp_norm norm=" << norm
                              << " step=" << step
                              << "\n";
                }
            }

            if (step > adjustment_step_min) {
                T p_new = (price_scale * (norm - step) + step * target_price) / norm;

                auto xp_new = xp;
                xp_new[1] = xp[1] * p_new / price_scale;

                T D_new   = MathOps<T>::newton_D(
                    _A_gamma[0], _A_gamma[1], xp_new, 0
                );
                T new_xcp = _xcp(D_new, p_new);
                T new_vp  = (total_supply > PoolTraits<T>::ZERO())
                    ? (PoolTraits<T>::PRECISION() * new_xcp / total_supply)
                    : PoolTraits<T>::PRECISION();

                // Optional burn to hit goal virtual_price
                T burn    = PoolTraits<T>::ZERO();
                T goal_vp = PoolTraits<T>::max(lp_xcp_profit, vp);
                if (new_vp < goal_vp) {
                    T tweaked_supply = (PoolTraits<T>::PRECISION() * new_xcp) / goal_vp;
                    if (tweaked_supply < total_supply) {
                        T diff      = total_supply - tweaked_supply;
                        T unlocked2 = _donation_shares();
                        burn        = (diff < unlocked2) ? diff : unlocked2;
                        if (total_supply > burn) {
                            new_vp = (
                                PoolTraits<T>::PRECISION() * new_xcp
                            ) / (total_supply - burn);
                        }
                    }
                }

                if (trace) {
                    if constexpr (std::is_same_v<T, uint256>) {
                        std::cout << "TRACE tp_candidate p_new=" << p_new.template convert_to<std::string>()
                                  << " new_vp=" << new_vp.template convert_to<std::string>()
                                  << " burn=" << burn.template convert_to<std::string>()
                                  << "\n";
                    } else {
                        std::cout << "TRACE tp_candidate p_new=" << p_new
                                  << " new_vp=" << new_vp
                                  << " burn=" << burn
                                  << "\n";
                    }
                }

                // Commit if within allowed region
                if (new_vp > PoolTraits<T>::PRECISION() && new_vp >= lp_xcp_profit) {
                    D = D_new;
                    virtual_price      = new_vp;
                    cached_price_scale = p_new;
                    if (burn > PoolTraits<T>::ZERO()) {
                        T shares_unlocked  = _donation_shares(false);
                        T shares_available = _donation_shares(true);

                        T shares_unlocked_new = shares_unlocked;
                        if (shares_available > PoolTraits<T>::ZERO()) {
                            shares_unlocked_new = shares_unlocked - (burn * shares_unlocked) / shares_available;
                        }

                        T new_total = donation_shares - burn;
                        T new_elapsed = PoolTraits<T>::ZERO();
                        if (new_total > PoolTraits<T>::ZERO() && shares_unlocked_new > PoolTraits<T>::ZERO()) {
                            new_elapsed = (shares_unlocked_new * donation_duration) / new_total;
                        }

                        donation_shares = new_total;
                        totalSupply    -= burn;
                        last_donation_release_ts = T(block_timestamp) - new_elapsed;
                    }
                    if (trace) {
                        std::cout << "TRACE tp_commit price_scale=";
                        if constexpr (std::is_same_v<T, uint256>) {
                            std::cout << cached_price_scale.template convert_to<std::string>();
                        } else {
                            std::cout << cached_price_scale;
                        }
                        std::cout << "\n";
                    }
                    _update_policy_state(xp_new, p_new, price_oracle, last_prices, new_vp, xcp_profit, D_new);
                    return p_new;
                }
            }
        }

        D = _D;
        virtual_price = vp;
        _update_policy_state(xp, price_scale, price_oracle, last_prices, virtual_price, xcp_profit, D);
        return price_scale;
    }

    // Views
    T donation_unlocked(bool donation_protection = true) const {
        return _donation_shares(donation_protection);
    }

    T get_virtual_price() const {
        return (virtual_price == Traits::ZERO()) ? Traits::PRECISION() : virtual_price;
    }

    T get_p() const {
        if (balances[0] == Traits::ZERO() || balances[1] == Traits::ZERO()) {
            return cached_price_scale;
        }
        return last_prices;
    }

    T get_vp_boosted() const {
        T xcp = _xcp(D, cached_price_scale);
        T donation_unlocked = _donation_shares();
        T locked_supply     = totalSupply - donation_unlocked;
        return (locked_supply == Traits::ZERO()) ? Traits::PRECISION() : (PoolTraits<T>::PRECISION() * xcp / locked_supply);
    }

    // Testing helpers
    void set_block_timestamp(uint64_t ts) {
        block_timestamp = ts;
        if (D == Traits::ZERO() && totalSupply == Traits::ZERO()) {
            last_timestamp = ts;
        }
    }

    void advance_time(uint64_t seconds) {
        block_timestamp += seconds;
    }
};

// Convenience aliases
using TwoCryptoPoolI = TwoCryptoPool<uint256>;
using TwoCryptoPoolD = TwoCryptoPool<double>;

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
