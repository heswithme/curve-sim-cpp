from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_policy_fee_api_requires_xp_and_price_scale_api_is_no_arg() -> None:
    policy_hpp = (ROOT / "cpp_modular/include/pools/twocrypto_fx/policy.hpp").read_text()
    pool_hpp = (ROOT / "cpp_modular/include/pools/twocrypto_fx/twocrypto.hpp").read_text()

    assert "T get_fee(const std::array<T, 2>& xp) const" in policy_hpp
    assert "T get_price_scale() const" in policy_hpp

    for helper in (
        "fixed_fee",
        "zero_stub_fee",
        "oracle_x2_sequential_fee",
        "twocrypto_policy_fee",
    ):
        assert f"T {helper}(const std::array<T, 2>& xp) const" in policy_hpp

    assert "policy.get_fee(xp)" in pool_hpp
    assert "policy.get_fee()" not in pool_hpp
    assert "policy.get_price_scale()" in pool_hpp
