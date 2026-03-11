// Runtime-dispatched runner for arb_harness and eval_server
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/json.hpp>

#include "core/common.hpp"
#include "core/json_utils.hpp"
#include "events/types.hpp"
#include "harness/donation.hpp"
#include "harness/idle_tick.hpp"
#include "harness/logging.hpp"
#include "harness/metrics.hpp"
#include "harness/runtime_backend.hpp"
#include "harness/user_swap.hpp"
#include "pools/config.hpp"
#include "pools/twocrypto_fx/helpers.hpp"
#include "trading/arbitrageur.hpp"
#include "trading/costs.hpp"
#include "trading/cowswap_trader.hpp"

namespace arb {
namespace harness {

using RuntimePoolConfig = std::pair<pools::PoolInit<double>, trading::Costs<double>>;

struct RuntimeRunConfig {
    double min_swap_frac{1e-6};
    double max_swap_frac{1.0};
    uint64_t dustswap_freq_s{3600};
    uint64_t user_swap_freq_s{0};
    double user_swap_size_frac{0.01};
    double user_swap_thresh{0.05};
    bool save_actions{false};
    bool detailed_log{false};
    size_t detailed_interval{1};
    bool enable_slippage_probes{true};
    std::string cowswap_path;
    double cowswap_fee_bps{0.0};
};

struct RuntimePoolResult {
    std::string tag;

    Metrics<double> metrics{};
    TimeWeightedMetrics<double> tw_metrics{};
    SlippageProbes<double> slippage_probes{};

    uint64_t t_start{0};
    uint64_t t_end{0};
    double tvl_start{0.0};
    double donation_apy{0.0};
    double donation_frequency{0.0};

    std::array<double, 2> balances{0.0, 0.0};
    double D{0.0};
    double totalSupply{0.0};
    double price_scale{0.0};
    double price_oracle{0.0};
    double virtual_price{0.0};
    double xcp_profit{0.0};
    double vp_boosted{0.0};
    double donation_shares{0.0};
    double donation_unlocked{0.0};
    double last_prices{0.0};
    uint64_t timestamp{0};

    boost::json::object echo_pool{};
    boost::json::object echo_costs{};
    boost::json::object final_state_json{};

    std::vector<Action<double>> actions{};
    std::vector<DetailedEntry<double>> detailed_entries{};

    double elapsed_ms{0.0};
    bool success{false};
    std::string error_msg;

    double duration_s() const {
        return (t_end > t_start) ? static_cast<double>(t_end - t_start) : 0.0;
    }
};

inline std::string integer_string_from_double_round(double value) {
    if (!std::isfinite(value) || value <= 0.0) {
        return "0";
    }
    const long double rounded = std::floor(static_cast<long double>(value) + 0.5L);
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(0);
    oss << rounded;
    return oss.str();
}

inline std::string scaled_1e18_string_from_double_floor(double value) {
    if (!std::isfinite(value) || value <= 0.0) {
        return "0";
    }
    const long double scaled = std::floor(static_cast<long double>(value) * WAD);
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(0);
    oss << scaled;
    return oss.str();
}

inline std::string fee_1e10_string_from_double_floor(double value) {
    if (!std::isfinite(value) || value <= 0.0) {
        return "0";
    }
    const long double scaled = std::floor(static_cast<long double>(value) * FEE_SCALE);
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(0);
    oss << scaled;
    return oss.str();
}

inline void set_pool_plain_json(boost::json::object& pool, const char* key, double value) {
    pool[key] = integer_string_from_double_round(value);
}

inline void set_pool_wad_json(boost::json::object& pool, const char* key, double value) {
    pool[key] = scaled_1e18_string_from_double_floor(value);
}

inline void set_pool_fee_json(boost::json::object& pool, const char* key, double value) {
    pool[key] = fee_1e10_string_from_double_floor(value);
}

inline std::string format_duration(double seconds) {
    int total_s = static_cast<int>(seconds + 0.5);
    int h = total_s / 3600;
    int m = (total_s % 3600) / 60;
    int s = total_s % 60;

    std::ostringstream oss;
    if (h > 0) {
        oss << h << ":" << std::setfill('0') << std::setw(2) << m << ":" << std::setw(2) << s;
    } else {
        oss << m << ":" << std::setfill('0') << std::setw(2) << s;
    }
    return oss.str();
}

namespace detail {

using U256 = pools::twocrypto_fx::uint256;
using DoublePool = pools::twocrypto_fx::TwoCryptoPool<double>;
using LongDoublePool = pools::twocrypto_fx::TwoCryptoPool<long double>;
using UIntPool = pools::twocrypto_fx::TwoCryptoPool<U256>;

struct ExchangeExec {
    bool success{false};
    double dx_actual{0.0};
    double dy_after_fee{0.0};
    double fee_tokens{0.0};
};

struct DonationExec {
    bool success{false};
    double amt0_actual{0.0};
    double amt1_actual{0.0};
};

inline double initial_price_or_default(
    const pools::PoolInit<double>& pool_init,
    const std::vector<Event>& events
) {
    double initial_price = pool_init.initial_price;
    if (!(initial_price > 0.0) && !events.empty()) {
        initial_price = static_cast<double>(events.front().p_cex);
    }
    if (!(initial_price > 0.0)) {
        initial_price = 1.0;
    }
    return initial_price;
}

inline uint64_t initial_timestamp_or_default(
    const pools::PoolInit<double>& pool_init,
    const std::vector<Event>& events
) {
    uint64_t init_ts = pool_init.start_ts;
    if (init_ts == 0 && !events.empty()) {
        init_ts = events.front().ts;
    }
    if (init_ts == 0) {
        init_ts = 1700000000ULL;
    }
    return init_ts;
}

inline std::array<double, 2> initial_liquidity_or_default(
    const pools::PoolInit<double>& pool_init,
    double initial_price
) {
    std::array<double, 2> liq = pool_init.initial_liq;
    if (!(liq[0] > 0.0) || !(liq[1] > 0.0)) {
        liq[0] = 1000000.0;
        liq[1] = liq[0] / initial_price;
    }
    return liq;
}

inline double u256_to_double_unscaled(const U256& value) {
    return static_cast<double>(value.convert_to<long double>());
}

inline U256 u256_from_integer_string(const std::string& s) {
    if (s.empty()) {
        return U256(0);
    }
    return U256(s);
}

inline std::string integer_string_from_json_value(const boost::json::value& value) {
    if (value.is_string()) {
        const std::string raw = std::string(value.as_string().c_str());
        if (!raw.empty() && raw.find_first_not_of("0123456789") == std::string::npos) {
            return raw;
        }
        try {
            return integer_string_from_double_round(std::stod(raw));
        } catch (...) {
            return "0";
        }
    }
    if (value.is_uint64()) {
        return std::to_string(value.as_uint64());
    }
    if (value.is_int64()) {
        return std::to_string(std::max<int64_t>(0, value.as_int64()));
    }
    if (value.is_double()) {
        return integer_string_from_double_round(value.as_double());
    }
    return "0";
}

inline U256 wad_from_double_floor(double value) {
    return u256_from_integer_string(scaled_1e18_string_from_double_floor(value));
}

inline U256 fee_from_double_floor(double value) {
    return u256_from_integer_string(fee_1e10_string_from_double_floor(value));
}

inline U256 integer_from_double_round(double value) {
    return u256_from_integer_string(integer_string_from_double_round(value));
}

inline U256 parse_u256_json_field(
    const boost::json::object& obj,
    const char* key,
    const U256& fallback
) {
    if (const auto* v = obj.if_contains(key)) {
        return u256_from_integer_string(integer_string_from_json_value(*v));
    }
    return fallback;
}

inline U256 parse_u256_initial_liq(
    const boost::json::object& obj,
    size_t index,
    const U256& fallback
) {
    if (const auto* v = obj.if_contains("initial_liquidity")) {
        if (v->is_array()) {
            const auto& arr = v->as_array();
            if (index < arr.size()) {
                return u256_from_integer_string(integer_string_from_json_value(arr[index]));
            }
        }
    }
    return fallback;
}

template <typename Real>
class FloatingBackend {
public:
    using Pool = pools::twocrypto_fx::TwoCryptoPool<Real>;

    FloatingBackend(const pools::PoolInit<double>& pool_init, const std::vector<Event>& events)
        : pool_(
            {
                static_cast<Real>(pool_init.precisions[0]),
                static_cast<Real>(pool_init.precisions[1]),
            },
            static_cast<Real>(pool_init.A),
            static_cast<Real>(pool_init.gamma),
            static_cast<Real>(pool_init.mid_fee),
            static_cast<Real>(pool_init.out_fee),
            static_cast<Real>(pool_init.fee_gamma),
            static_cast<Real>(pool_init.allowed_extra_profit),
            static_cast<Real>(pool_init.adjustment_step),
            static_cast<Real>(pool_init.ma_time),
            static_cast<Real>(initial_price_or_default(pool_init, events))
        ) {
        const double initial_price = initial_price_or_default(pool_init, events);

        pool_.set_block_timestamp(initial_timestamp_or_default(pool_init, events));
        const auto liq = initial_liquidity_or_default(pool_init, initial_price);
        pool_.add_liquidity(
            {
                static_cast<Real>(liq[0]),
                static_cast<Real>(liq[1]),
            },
            static_cast<Real>(0)
        );
    }

    DoublePool snapshot() const {
        if constexpr (std::is_same_v<Real, double>) {
            return pool_;
        } else {
            return pools::twocrypto_fx::make_runtime_view(pool_);
        }
    }

    void set_block_timestamp(uint64_t ts) {
        pool_.set_block_timestamp(ts);
    }

    uint64_t block_timestamp() const {
        return pool_.block_timestamp;
    }

    bool add_donation(double amt0_target, double amt1_target, DonationExec& out) {
        const Real amt0 = static_cast<Real>(std::max(0.0, amt0_target));
        const Real amt1 = static_cast<Real>(std::max(0.0, amt1_target));
        if (!(amt0 > Real(0)) && !(amt1 > Real(0))) {
            return false;
        }
        try {
            pool_.add_liquidity({amt0, amt1}, Real(0), true);
            out.success = true;
            out.amt0_actual = static_cast<double>(amt0);
            out.amt1_actual = static_cast<double>(amt1);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool exchange(size_t i, size_t j, double dx_target, ExchangeExec& out) {
        if (!(dx_target > 0.0)) {
            return false;
        }
        const Real dx = static_cast<Real>(dx_target);
        try {
            const auto res = pool_.exchange(
                static_cast<Real>(i),
                static_cast<Real>(j),
                dx,
                Real(0)
            );
            out.success = true;
            out.dx_actual = static_cast<double>(dx);
            out.dy_after_fee = static_cast<double>(res[0]);
            out.fee_tokens = static_cast<double>(res[1]);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool tick() {
        try {
            pool_.tick();
            return true;
        } catch (...) {
            return false;
        }
    }

    boost::json::object final_state_json() const {
        return pools::twocrypto_fx::runtime_final_state_json(pool_);
    }

private:
    Pool pool_;
};

class UintBackend {
public:
    UintBackend(const pools::PoolInit<double>& pool_init, const std::vector<Event>& events)
        : pool_([&]() {
            const auto& raw_pool = pool_init.echo_pool;
            U256 initial_price = parse_u256_json_field(
                raw_pool,
                "initial_price",
                wad_from_double_floor(initial_price_or_default(pool_init, events))
            );
            if (initial_price == U256(0)) {
                initial_price = wad_from_double_floor(1.0);
            }

            return UIntPool(
                {U256(1), U256(1)},
                parse_u256_json_field(raw_pool, "A", integer_from_double_round(pool_init.A)),
                parse_u256_json_field(raw_pool, "gamma", wad_from_double_floor(pool_init.gamma)),
                parse_u256_json_field(raw_pool, "mid_fee", fee_from_double_floor(pool_init.mid_fee)),
                parse_u256_json_field(raw_pool, "out_fee", fee_from_double_floor(pool_init.out_fee)),
                parse_u256_json_field(raw_pool, "fee_gamma", wad_from_double_floor(pool_init.fee_gamma)),
                parse_u256_json_field(raw_pool, "allowed_extra_profit", wad_from_double_floor(pool_init.allowed_extra_profit)),
                parse_u256_json_field(raw_pool, "adjustment_step", wad_from_double_floor(pool_init.adjustment_step)),
                parse_u256_json_field(raw_pool, "ma_time", integer_from_double_round(pool_init.ma_time)),
                initial_price
            );
        }()) {
        const auto& raw_pool = pool_init.echo_pool;

        pool_.set_block_timestamp(initial_timestamp_or_default(pool_init, events));

        U256 liq0 = parse_u256_initial_liq(raw_pool, 0, wad_from_double_floor(pool_init.initial_liq[0]));
        U256 liq1 = parse_u256_initial_liq(raw_pool, 1, wad_from_double_floor(pool_init.initial_liq[1]));
        if (liq0 == U256(0) || liq1 == U256(0)) {
            const U256 raw_initial_price = parse_u256_json_field(
                raw_pool,
                "initial_price",
                wad_from_double_floor(initial_price_or_default(pool_init, events))
            );
            const double fallback_price = std::max(
                1e-18,
                pools::twocrypto_fx::runtime_wad_to_double(
                    raw_initial_price == U256(0) ? wad_from_double_floor(1.0) : raw_initial_price
                )
            );
            liq0 = wad_from_double_floor(1000000.0);
            liq1 = wad_from_double_floor(1000000.0 / fallback_price);
        }

        pool_.add_liquidity({liq0, liq1}, U256(0));
    }

    DoublePool snapshot() const {
        return pools::twocrypto_fx::make_runtime_view(pool_);
    }

    void set_block_timestamp(uint64_t ts) {
        pool_.set_block_timestamp(ts);
    }

    uint64_t block_timestamp() const {
        return pool_.block_timestamp;
    }

    bool add_donation(double amt0_target, double amt1_target, DonationExec& out) {
        const U256 amt0 = wad_from_double_floor(std::max(0.0, amt0_target));
        const U256 amt1 = wad_from_double_floor(std::max(0.0, amt1_target));
        if (amt0 == U256(0) && amt1 == U256(0)) {
            return false;
        }
        try {
            pool_.add_liquidity({amt0, amt1}, U256(0), true);
            out.success = true;
            out.amt0_actual = pools::twocrypto_fx::runtime_wad_to_double(amt0);
            out.amt1_actual = pools::twocrypto_fx::runtime_wad_to_double(amt1);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool exchange(size_t i, size_t j, double dx_target, ExchangeExec& out) {
        const U256 dx = wad_from_double_floor(dx_target);
        if (dx == U256(0)) {
            return false;
        }
        try {
            const auto res = pool_.exchange(U256(i), U256(j), dx, U256(0));
            out.success = true;
            out.dx_actual = pools::twocrypto_fx::runtime_wad_to_double(dx);
            out.dy_after_fee = pools::twocrypto_fx::runtime_wad_to_double(res[0]);
            out.fee_tokens = pools::twocrypto_fx::runtime_wad_to_double(res[1]);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool tick() {
        try {
            pool_.tick();
            return true;
        } catch (...) {
            return false;
        }
    }

    boost::json::object final_state_json() const {
        return pools::twocrypto_fx::runtime_final_state_json(pool_);
    }

private:
    UIntPool pool_;
};

template <typename Backend>
RuntimePoolResult run_single_pool_backend(
    const pools::PoolInit<double>& pool_init,
    const trading::Costs<double>& costs,
    const std::vector<Event>& events,
    const RuntimeRunConfig& cfg,
    const std::vector<trading::CowswapTrade>* cowswap_trades = nullptr,
    const std::vector<Candle>* candles = nullptr
) {
    RuntimePoolResult result;
    result.tag = pool_init.tag;
    result.echo_pool = pool_init.echo_pool;
    result.echo_costs = pool_init.echo_costs;

    auto t_start = std::chrono::high_resolution_clock::now();

    try {
        Backend backend(pool_init, events);

        EventLoopResult<double> loop_result{};
        Metrics<double>& m = loop_result.metrics;
        TimeWeightedMetrics<double>& tw = loop_result.tw_metrics;
        SlippageProbes<double>& sp = loop_result.slippage_probes;

        const size_t n_events = events.size();
        if (n_events > 0) {
            loop_result.t_start = events.front().ts;
            loop_result.t_end = events.back().ts;
            const DoublePool initial_view = backend.snapshot();
            loop_result.tvl_start =
                initial_view.balances[0] +
                initial_view.balances[1] * initial_view.cached_price_scale;
        }
        loop_result.donation_apy = pool_init.donation_apy;

        std::array<double, SlippageProbes<double>::N_SIZES> probe_sizes_coin0{};
        if (cfg.enable_slippage_probes) {
            for (size_t k = 0; k < SlippageProbes<double>::N_SIZES; ++k) {
                probe_sizes_coin0[k] =
                    loop_result.tvl_start * SlippageProbes<double>::SIZE_FRACS[k];
            }
        }

        DonationCfg<double> dcfg{};
        if (pool_init.donation_apy > 0.0 && pool_init.donation_frequency > 0.0 && n_events > 0) {
            const DoublePool initial_view = backend.snapshot();
            dcfg.init(
                pool_init.donation_apy,
                pool_init.donation_frequency,
                pool_init.donation_coins_ratio,
                initial_timestamp_or_default(pool_init, events),
                initial_view
            );
        }

        IdleTickCfg<double> icfg{};
        icfg.freq_s = cfg.dustswap_freq_s;

        UserSwapCfg<double> ucfg{};
        ucfg.freq_s = cfg.user_swap_freq_s;
        ucfg.size_frac = cfg.user_swap_size_frac;
        ucfg.thresh = cfg.user_swap_thresh;
        if (ucfg.enabled() && n_events > 0) {
            ucfg.init(initial_timestamp_or_default(pool_init, events));
        }

        ActionLogger<double> action_logger(cfg.save_actions);
        DetailedLogger<double> detailed_logger(cfg.detailed_log, cfg.detailed_interval);
        if (detailed_logger.enabled() && candles == nullptr) {
            throw std::invalid_argument("detailed_log enabled but candles were not provided");
        }

        size_t cowswap_idx = 0;
        if (cowswap_trades && !cowswap_trades->empty() && n_events > 0) {
            auto it = std::lower_bound(
                cowswap_trades->begin(),
                cowswap_trades->end(),
                events.front().ts,
                [](const trading::CowswapTrade& trade, uint64_t ts) {
                    return trade.ts < ts;
                }
            );
            cowswap_idx = static_cast<size_t>(it - cowswap_trades->begin());
        }

        auto sample_slippage_probes = [&](uint64_t ts, double p_cex) {
            if (!cfg.enable_slippage_probes || !(p_cex > 0.0)) {
                return;
            }

            const DoublePool view = backend.snapshot();
            for (size_t k = 0; k < SlippageProbes<double>::N_SIZES; ++k) {
                sp.accumulate_previous(k, ts);

                const double S = probe_sizes_coin0[k];
                const auto pr01 = pools::twocrypto_fx::simulate_exchange_once(view, 0, 1, S);
                const double ideal1 = S / p_cex;
                double s01 = 0.0;
                if (ideal1 > 0.0) {
                    s01 = 1.0 - (pr01.first / ideal1);
                }

                const double dx1 = S / p_cex;
                const auto pr10 = pools::twocrypto_fx::simulate_exchange_once(view, 1, 0, dx1);
                double s10 = 0.0;
                if (S > 0.0) {
                    s10 = 1.0 - (pr10.first / S);
                }
                sp.sample(k, ts, s01, s10);
            }
        };

        auto sample_pre_trade = [&](uint64_t ts, double cex_price, const DoublePool& view) {
            tw.sample_price_error(ts, view.cached_price_scale, cex_price);

            const double x0p = view.balances[0];
            const double x1p = view.balances[1] * cex_price;
            tw.sample_imbalance(ts, x0p, x1p);

            const auto xp_now = pools::twocrypto_fx::pool_xp_current(view);
            const double cur_fee = pools::twocrypto_fx::dyn_fee(
                xp_now, view.mid_fee, view.out_fee, view.fee_gamma
            );
            tw.sample_fee(ts, cur_fee);
        };

        auto apply_donation = [&](uint64_t ts, const DoublePool& before_view) {
            if (!dcfg.enabled || dcfg.next_ts == 0 || ts < dcfg.next_ts) {
                return;
            }

            constexpr double SEC_PER_YEAR = 365.0 * 86400.0;
            const double po = before_view.cached_price_oracle;
            const double frac = dcfg.apy * (static_cast<double>(dcfg.freq_s) / SEC_PER_YEAR);
            const double tvl = dcfg.use_base_tvl
                ? dcfg.base_tvl
                : (before_view.balances[0] + before_view.balances[1] * po);
            const uint64_t ts_due = dcfg.next_ts;

            DonationExec exec{};
            const double amt0_target = (1.0 - dcfg.ratio1) * tvl * frac;
            const double amt1_target = (po > 0.0) ? (dcfg.ratio1 * tvl * frac / po) : 0.0;
            if (backend.add_donation(amt0_target, amt1_target, exec)) {
                const DoublePool after_view = backend.snapshot();
                if (differs_rel(after_view.cached_price_scale, before_view.cached_price_scale)) {
                    m.n_rebalances += 1;
                }

                m.donations += 1;
                m.donation_amounts_total[0] += exec.amt0_actual;
                m.donation_amounts_total[1] += exec.amt1_actual;
                m.donation_coin0_total +=
                    exec.amt0_actual + exec.amt1_actual * before_view.cached_price_scale;

                if (action_logger.enabled()) {
                    DonationResult<double> don_res{};
                    don_res.success = true;
                    don_res.ts_due = ts_due;
                    don_res.amounts = {exec.amt0_actual, exec.amt1_actual};
                    don_res.price_scale = before_view.cached_price_scale;
                    action_logger.log_donation(ts, don_res, dcfg);
                }
            }

            dcfg.next_ts = ts_due + dcfg.freq_s;
        };

        auto apply_user_swap = [&](uint64_t ts, double cex_price) {
            if (!ucfg.enabled()) return;
            if (ucfg.next_ts == 0 || ts < ucfg.next_ts) return;

            ucfg.next_ts += ucfg.freq_s;
            if (!(cex_price > 0.0)) return;

            const DoublePool view = backend.snapshot();
            const double spot = view.get_p();
            if (!(spot > 0.0)) return;

            const double rel = arb::abs_value(spot / cex_price - 1.0);
            if (rel > ucfg.thresh) return;

            const size_t i_from = ucfg.next_dir & 1;
            const size_t j_to = i_from ^ 1;
            const double bal_from = view.balances[i_from];
            if (!(bal_from > 0.0)) return;

            double frac = ucfg.size_frac;
            if (frac > 1.0) frac = 1.0;
            if (!(frac > 0.0)) return;

            ExchangeExec exec{};
            if (backend.exchange(i_from, j_to, bal_from * frac, exec)) {
                ucfg.next_dir ^= 1;
            }
        };

        auto execute_arb = [&](const Event& ev, double cex_price) -> bool {
            const DoublePool before_view = backend.snapshot();

            double volume_cap = std::numeric_limits<double>::infinity();
            if (costs.use_volume_cap) {
                volume_cap = static_cast<double>(ev.volume) * costs.volume_cap_mult;
                if (!costs.volume_cap_is_coin1) {
                    volume_cap *= cex_price;
                }
            }

            const auto dec = trading::decide_trade(
                before_view,
                cex_price,
                costs,
                volume_cap,
                cfg.min_swap_frac,
                cfg.max_swap_frac
            );
            if (!dec.do_trade) {
                return false;
            }

            ExchangeExec exec{};
            if (!backend.exchange(static_cast<size_t>(dec.i), static_cast<size_t>(dec.j), dec.dx, exec)) {
                return false;
            }

            const DoublePool after_view = backend.snapshot();
            const double notional_coin0 = (dec.i == 0)
                ? exec.dx_actual
                : exec.dx_actual * cex_price;
            const double fee_cex = costs.arb_fee_bps / 10000.0;
            const double profit_coin0 = (dec.i == 0)
                ? (exec.dy_after_fee * cex_price * (1.0 - fee_cex) - exec.dx_actual - costs.gas_coin0)
                : (exec.dy_after_fee - exec.dx_actual * cex_price * (1.0 + fee_cex) - costs.gas_coin0);

            m.trades += 1;
            m.notional += notional_coin0;
            m.lp_fee_coin0 += (dec.j == 1 ? exec.fee_tokens * cex_price : exec.fee_tokens);
            m.arb_pnl_coin0 += profit_coin0;

            const double gross_dy_tokens = exec.dy_after_fee + exec.fee_tokens;
            if (gross_dy_tokens > 0.0 && notional_coin0 > 0.0) {
                const double fee_frac = exec.fee_tokens / gross_dy_tokens;
                m.fee_wsum += fee_frac * notional_coin0;
                m.fee_w += notional_coin0;
            }

            if (differs_rel(after_view.cached_price_scale, before_view.cached_price_scale)) {
                m.n_rebalances += 1;
            }

            sample_slippage_probes(ev.ts, cex_price);
            action_logger.log_exchange(
                ev.ts,
                dec.i,
                dec.j,
                exec.dx_actual,
                exec.dy_after_fee,
                exec.fee_tokens,
                profit_coin0,
                cex_price,
                before_view.get_p(),
                before_view.cached_price_oracle,
                before_view.cached_price_scale,
                before_view.last_timestamp,
                before_view.last_prices,
                before_view.xcp_profit,
                before_view.get_vp_boosted(),
                after_view,
                tw
            );
            return true;
        };

        auto apply_cowswap = [&]() -> bool {
            if (!(cowswap_trades && cowswap_idx < cowswap_trades->size())) {
                return false;
            }

            bool executed_any = false;
            while (cowswap_idx < cowswap_trades->size() && (*cowswap_trades)[cowswap_idx].ts <= backend.block_timestamp()) {
                const auto& trade = (*cowswap_trades)[cowswap_idx++];
                const DoublePool before_view = backend.snapshot();

                const double dx_target = trade.is_buy ? trade.usd_amount : trade.wbtc_amount;
                const double required_dy = trade.is_buy ? trade.wbtc_amount : trade.usd_amount;
                const auto sim = trade.is_buy
                    ? pools::twocrypto_fx::simulate_exchange_once(before_view, 0, 1, dx_target)
                    : pools::twocrypto_fx::simulate_exchange_once(before_view, 1, 0, dx_target);
                const double threshold_dy = required_dy * (1.0 + cfg.cowswap_fee_bps / 10000.0);
                if (sim.first < threshold_dy) {
                    m.cowswap_skipped += 1;
                    continue;
                }

                ExchangeExec exec{};
                if (!backend.exchange(trade.is_buy ? 0 : 1, trade.is_buy ? 1 : 0, dx_target, exec)) {
                    m.cowswap_skipped += 1;
                    continue;
                }

                const DoublePool after_view = backend.snapshot();
                const double advantage_bps = (required_dy > 0.0)
                    ? (sim.first / required_dy - 1.0) * 10000.0
                    : 0.0;

                m.cowswap_trades += 1;
                m.cowswap_notional_coin0 += trade.is_buy
                    ? exec.dx_actual
                    : exec.dx_actual * after_view.cached_price_scale;
                m.cowswap_lp_fee_coin0 += trade.is_buy
                    ? exec.fee_tokens * after_view.cached_price_scale
                    : exec.fee_tokens;

                if (action_logger.enabled()) {
                    action_logger.log_cowswap(
                        trade.ts,
                        trade.is_buy,
                        exec.dx_actual,
                        exec.dy_after_fee,
                        exec.fee_tokens,
                        required_dy,
                        advantage_bps,
                        cfg.cowswap_fee_bps,
                        before_view.cached_price_scale,
                        after_view.cached_price_scale
                    );
                }
                executed_any = true;
            }

            return executed_any;
        };

        auto apply_idle_tick = [&](uint64_t ts, double cex_price) -> bool {
            const DoublePool before_view = backend.snapshot();
            if (ts < before_view.last_timestamp + icfg.freq_s) {
                return false;
            }
            if (!backend.tick()) {
                return false;
            }

            const DoublePool after_view = backend.snapshot();
            if (differs_rel(after_view.cached_price_scale, before_view.cached_price_scale)) {
                m.n_rebalances += 1;
            }
            sample_slippage_probes(ts, cex_price);
            action_logger.log_tick(
                ts,
                cex_price,
                before_view.cached_price_scale,
                before_view.cached_price_oracle,
                before_view.xcp_profit,
                before_view.get_vp_boosted(),
                after_view
            );
            return true;
        };

        for (size_t ev_idx = 0; ev_idx < n_events; ++ev_idx) {
            const auto& ev = events[ev_idx];
            backend.set_block_timestamp(ev.ts);
            const double cex_price = static_cast<double>(ev.p_cex);

            const DoublePool before_view = backend.snapshot();
            sample_pre_trade(ev.ts, cex_price, before_view);
            apply_donation(ev.ts, before_view);

            if (!(cex_price > 0.0)) {
                continue;
            }

            apply_user_swap(ev.ts, cex_price);

            bool did_any_trade = execute_arb(ev, cex_price);
            if (apply_cowswap()) {
                did_any_trade = true;
            }
            if (!did_any_trade) {
                did_any_trade = apply_idle_tick(ev.ts, cex_price);
            }

            if (detailed_logger.enabled()) {
                const size_t candle_idx = static_cast<size_t>(ev.candle_idx);
                if (candle_idx >= candles->size()) {
                    throw std::out_of_range("event.candle_idx out of range");
                }
                const DoublePool after_view = backend.snapshot();
                detailed_logger.log_event(
                    after_view,
                    ev.ts,
                    (*candles)[candle_idx],
                    cex_price,
                    m.trades,
                    m.n_rebalances
                );
            }
        }

        result.metrics = loop_result.metrics;
        result.tw_metrics = loop_result.tw_metrics;
        result.slippage_probes = loop_result.slippage_probes;
        result.t_start = loop_result.t_start;
        result.t_end = loop_result.t_end;
        result.tvl_start = loop_result.tvl_start;
        result.donation_apy = loop_result.donation_apy;
        result.donation_frequency = pool_init.donation_frequency;
        result.actions = action_logger.take_actions();
        result.detailed_entries = detailed_logger.take_entries();

        const DoublePool final_view = backend.snapshot();
        result.balances = final_view.balances;
        result.D = final_view.D;
        result.totalSupply = final_view.totalSupply;
        result.price_scale = final_view.cached_price_scale;
        result.price_oracle = final_view.cached_price_oracle;
        result.virtual_price = final_view.get_virtual_price();
        result.xcp_profit = final_view.xcp_profit;
        result.vp_boosted = final_view.get_vp_boosted();
        result.donation_shares = final_view.donation_shares;
        result.donation_unlocked = final_view.donation_unlocked();
        result.last_prices = final_view.last_prices;
        result.timestamp = final_view.block_timestamp;
        result.final_state_json = backend.final_state_json();
        result.success = true;
    } catch (const std::exception& e) {
        result.success = false;
        result.error_msg = e.what();
    } catch (...) {
        result.success = false;
        result.error_msg = "Unknown error";
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return result;
}

} // namespace detail

inline RuntimePoolResult run_single_pool_runtime(
    const pools::PoolInit<double>& pool_init,
    const trading::Costs<double>& costs,
    const std::vector<Event>& events,
    const RuntimeRunConfig& cfg,
    PoolBackend backend,
    const std::vector<trading::CowswapTrade>* cowswap_trades = nullptr,
    const std::vector<Candle>* candles = nullptr
) {
    switch (backend) {
        case PoolBackend::Double:
            return detail::run_single_pool_backend<detail::FloatingBackend<double>>(
                pool_init, costs, events, cfg, cowswap_trades, candles
            );
        case PoolBackend::LongDouble:
            return detail::run_single_pool_backend<detail::FloatingBackend<long double>>(
                pool_init, costs, events, cfg, cowswap_trades, candles
            );
        case PoolBackend::Uint:
            return detail::run_single_pool_backend<detail::UintBackend>(
                pool_init, costs, events, cfg, cowswap_trades, candles
            );
    }

    return detail::run_single_pool_backend<detail::FloatingBackend<double>>(
        pool_init, costs, events, cfg, cowswap_trades, candles
    );
}

inline std::vector<RuntimePoolResult> run_pools_parallel_runtime(
    const std::vector<RuntimePoolConfig>& pool_configs,
    const std::vector<Event>& events,
    const RuntimeRunConfig& cfg,
    PoolBackend backend,
    size_t n_threads = 0,
    bool verbose = true,
    const std::vector<Candle>* candles = nullptr
) {
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 1;
    }

    const size_t n_pools = pool_configs.size();
    std::vector<RuntimePoolResult> results(n_pools);
    if (n_pools == 0) {
        return results;
    }

    const size_t log_interval = std::max(size_t(1), std::min(n_pools / 100, size_t(1000)));

    std::vector<trading::CowswapTrade> cowswap_trades;
    const std::vector<trading::CowswapTrade>* cs_ptr = nullptr;
    if (!cfg.cowswap_path.empty()) {
        cowswap_trades = trading::load_cowswap_csv(cfg.cowswap_path);
        if (!cowswap_trades.empty()) {
            cs_ptr = &cowswap_trades;
            if (verbose) {
                std::lock_guard<std::mutex> lock(io_mu);
                std::cout << "loaded " << cowswap_trades.size()
                          << " cowswap trades from " << cfg.cowswap_path << "\n" << std::flush;
            }
        }
    }

    auto t_total_start = std::chrono::high_resolution_clock::now();

    if (n_pools == 1 || n_threads == 1) {
        for (size_t i = 0; i < n_pools; ++i) {
            const auto& [pool_init, costs] = pool_configs[i];
            results[i] = run_single_pool_runtime(pool_init, costs, events, cfg, backend, cs_ptr, candles);

            const size_t done = i + 1;
            if (verbose && (done % log_interval == 0 || done == n_pools)) {
                const auto now = std::chrono::high_resolution_clock::now();
                const double elapsed_s = std::chrono::duration<double>(now - t_total_start).count();
                const double avg_s = elapsed_s / done;
                const double eta_s = avg_s * (n_pools - done);

                std::lock_guard<std::mutex> lock(io_mu);
                std::cout << "pool " << done << "/" << n_pools
                          << " (" << (100 * done / n_pools) << "%)"
                          << " | elapsed:" << format_duration(elapsed_s)
                          << " | eta:" << format_duration(eta_s)
                          << "\n" << std::flush;
            }
        }
        return results;
    }

    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> completed{0};
    std::atomic<size_t> last_logged{0};

    auto worker = [&]() {
        while (true) {
            const size_t i = next_idx.fetch_add(1);
            if (i >= n_pools) break;

            const auto& [pool_init, costs] = pool_configs[i];
            results[i] = run_single_pool_runtime(pool_init, costs, events, cfg, backend, cs_ptr, candles);

            const size_t done = completed.fetch_add(1) + 1;
            if (verbose && (done == n_pools || done / log_interval > last_logged.load())) {
                last_logged.store(done / log_interval);

                const auto now = std::chrono::high_resolution_clock::now();
                const double elapsed_s = std::chrono::duration<double>(now - t_total_start).count();
                const double avg_s = elapsed_s / done;
                const double eta_s = avg_s * (n_pools - done);

                std::lock_guard<std::mutex> lock(io_mu);
                std::cout << "pool " << done << "/" << n_pools
                          << " (" << (100 * done / n_pools) << "%)"
                          << " | elapsed:" << format_duration(elapsed_s)
                          << " | eta:" << format_duration(eta_s)
                          << "\n" << std::flush;
            }
        }
    };

    const size_t actual_threads = std::min(n_threads, n_pools);
    std::vector<std::thread> threads;
    threads.reserve(actual_threads);
    for (size_t t = 0; t < actual_threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& th : threads) {
        th.join();
    }

    return results;
}

} // namespace harness
} // namespace arb
