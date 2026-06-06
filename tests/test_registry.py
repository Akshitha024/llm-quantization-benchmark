from __future__ import annotations

import pytest

from qs.recipes.registry import known, resolve


def test_known_includes_fp16_and_bnb_4bit() -> None:
    names = known()
    assert "fp16" in names
    assert "bnb_4bit" in names
    assert "bnb_nf4" in names


def test_resolve_returns_spec() -> None:
    s = resolve("bnb_nf4")
    assert s.bits == 4
    assert s.method == "bnb"
    assert ("bnb_4bit_quant_type", "nf4") in s.extras


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve("does_not_exist")


def test_fp16_is_lossless_method_none() -> None:
    s = resolve("fp16")
    assert s.method == "none"
    assert s.bits == 16
