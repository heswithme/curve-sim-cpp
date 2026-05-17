// Harness-facing facade over the uint256 twocrypto pool.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "harness/pool_value.hpp"
#include "pools/twocrypto_fx/helpers.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"

namespace arb {
namespace harness {

template <typename T>
class U256PoolAdapter {
public:
    using value_type = T;
    using RawT = pools::twocrypto_fx::uint256;
    using RawPool = pools::twocrypto_fx::TwoCryptoPool<RawT>;

    std::array<T, 2> balances{T(0), T(0)};
    std::array<T, 2> admin_balances{T(0), T(0)};
    T D{0};
    T totalSupply{0};
    T cached_price_scale{0};
    T cached_price_oracle{0};
    T virtual_price{0};
    T xcp_profit{0};
    T lp_xcp_profit{0};
    T donation_shares{0};
    T last_prices{0};
    uint64_t block_timestamp{0};
    uint64_t last_timestamp{0};

    U256PoolAdapter(
        const std::array<RawT, 2>& precisions,
        const RawT& A,
        const RawT& gamma,
        const RawT& mid_fee,
        const RawT& out_fee,
        const RawT& fee_gamma,
        const RawT& adjustment_step_min,
        const RawT& adjustment_step_max,
        const RawT& ma_time,
        const RawT& initial_price,
        const RawT& reserved_profit_fraction,
        const RawT& admin_fee,
        pools::twocrypto_fx::PolicyKind policy_kind,
        const pools::twocrypto_fx::PolicyConfig<RawT>& policy_config
    ) : raw_(
            precisions,
            A,
            gamma,
            mid_fee,
            out_fee,
            fee_gamma,
            adjustment_step_min,
            adjustment_step_max,
            ma_time,
            initial_price,
            reserved_profit_fraction,
            admin_fee,
            policy_kind,
            policy_config
        ) {
        sync();
    }

    RawPool& raw() { return raw_; }
    const RawPool& raw() const { return raw_; }

    void set_block_timestamp(uint64_t ts) {
        raw_.set_block_timestamp(ts);
        sync();
    }

    void advance_time(uint64_t seconds) {
        raw_.advance_time(seconds);
        sync();
    }

    void tick() {
        raw_.tick();
        sync();
    }

    T add_liquidity(const std::array<T, 2>& amounts, T min_mint, bool donation = false) {
        const auto minted = raw_.add_liquidity(
            {h_amount_to_pool<RawT>(amounts[0]), h_amount_to_pool<RawT>(amounts[1])},
            h_amount_to_pool<RawT>(min_mint),
            donation
        );
        sync();
        return pool_amount_to_h<T>(minted);
    }

    RawT add_liquidity(const std::array<RawT, 2>& amounts, RawT min_mint, bool donation = false) {
        const auto minted = raw_.add_liquidity(amounts, min_mint, donation);
        sync();
        return minted;
    }

    void set_donation_duration(const RawT& duration) {
        raw_.donation_duration = duration;
        sync();
    }

    std::array<T, 3> exchange(T i, T j, T dx, T min_dy) {
        const auto res = raw_.exchange(
            RawT(static_cast<size_t>(i)),
            RawT(static_cast<size_t>(j)),
            h_amount_to_pool_floor<RawT>(dx),
            h_amount_to_pool_floor<RawT>(min_dy)
        );
        sync();
        return {
            pool_amount_to_h<T>(res[0]),
            pool_amount_to_h<T>(res[1]),
            pool_price_to_h<T>(res[2])
        };
    }

    std::array<T, 3> exchange_from_preview(size_t i, size_t j, T dx, T, T) {
        return exchange(T(i), T(j), dx, T(0));
    }

    T get_virtual_price() const {
        return pool_price_to_h<T>(raw_.get_virtual_price());
    }

    T get_p() const {
        return pool_price_to_h<T>(raw_.get_p());
    }

    T get_vp_boosted() const {
        return pool_price_to_h<T>(raw_.get_vp_boosted());
    }

    T donation_unlocked(bool donation_protection = true) const {
        return pool_amount_to_h<T>(raw_.donation_unlocked(donation_protection));
    }

    T fee(const std::array<T, 2>&) const {
        return pool_fee_to_h<T>(raw_.fee(pools::twocrypto_fx::pool_xp_current(raw_)));
    }

    std::pair<T, T> simulate_exchange_once(size_t i, size_t j, T dx) const {
        const RawT dx_raw = h_amount_to_pool_floor<RawT>(dx);
        if (!(dx_raw > RawT(0))) {
            return {T(0), T(0)};
        }
        const auto res = pools::twocrypto_fx::simulate_exchange_once(raw_, i, j, dx_raw);
        return {pool_amount_to_h<T>(res.first), pool_amount_to_h<T>(res.second)};
    }

private:
    RawPool raw_;

    void sync() {
        balances[0] = pool_amount_to_h<T>(raw_.balances[0]);
        balances[1] = pool_amount_to_h<T>(raw_.balances[1]);
        admin_balances[0] = pool_amount_to_h<T>(raw_.admin_balances[0]);
        admin_balances[1] = pool_amount_to_h<T>(raw_.admin_balances[1]);
        D = pool_amount_to_h<T>(raw_.D);
        totalSupply = pool_amount_to_h<T>(raw_.totalSupply);
        cached_price_scale = pool_price_to_h<T>(raw_.cached_price_scale);
        cached_price_oracle = pool_price_to_h<T>(raw_.cached_price_oracle);
        virtual_price = pool_price_to_h<T>(raw_.get_virtual_price());
        xcp_profit = pool_price_to_h<T>(raw_.xcp_profit);
        lp_xcp_profit = pool_price_to_h<T>(raw_.lp_xcp_profit);
        donation_shares = pool_amount_to_h<T>(raw_.donation_shares);
        last_prices = pool_price_to_h<T>(raw_.last_prices);
        block_timestamp = raw_.block_timestamp;
        last_timestamp = raw_.last_timestamp;
    }
};

template <typename PoolT, typename T>
struct HarnessPoolFor {
    using type = pools::twocrypto_fx::TwoCryptoPool<T>;
};

template <typename T>
struct HarnessPoolFor<pools::twocrypto_fx::uint256, T> {
    using type = U256PoolAdapter<T>;
};

} // namespace harness
} // namespace arb

namespace arb {
namespace pools {
namespace twocrypto_fx {

template <typename T>
inline std::pair<T, T> simulate_exchange_once(
    const harness::U256PoolAdapter<T>& pool,
    size_t i,
    size_t j,
    T dx
) {
    return pool.simulate_exchange_once(i, j, dx);
}

template <typename T>
inline std::array<T, 2> pool_xp_current(const harness::U256PoolAdapter<T>& pool) {
    return {
        pool.balances[0],
        pool.balances[1] * pool.cached_price_scale
    };
}

template <typename T>
inline T balance_indicator(const harness::U256PoolAdapter<T>& pool) {
    const T denom = pool.balances[0] + pool.balances[1] * pool.cached_price_scale;
    if (!(denom > T(0))) {
        return T(0);
    }
    const T x0 = pool.balances[0];
    const T x1 = pool.balances[1] * pool.cached_price_scale;
    return T(4) * x0 * x1 / (denom * denom);
}

template <typename T>
inline T viewer_exchange_fee_fraction(const harness::U256PoolAdapter<T>& pool, T) {
    return pool.fee(pool_xp_current(pool));
}

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
