"""Quantization recipe specs.

Each recipe is a string the user passes on the CLI; we map to a config
that the loader turns into transformers.BitsAndBytesConfig / GPTQConfig /
AWQ etc. We intentionally do not ship the GPTQ/AWQ pre-quantized weights;
the loader will reach for them on the Hub the first time.
"""

from __future__ import annotations

from ..types import RecipeSpec

RECIPES: dict[str, RecipeSpec] = {
    "fp32": RecipeSpec(name="fp32", bits=32, method="none"),
    "fp16": RecipeSpec(name="fp16", bits=16, method="none"),
    "bnb_8bit": RecipeSpec(
        name="bnb_8bit", bits=8, method="bnb", extras=(("load_in_8bit", "True"),)
    ),
    "bnb_4bit": RecipeSpec(
        name="bnb_4bit",
        bits=4,
        method="bnb",
        extras=(("load_in_4bit", "True"), ("bnb_4bit_quant_type", "fp4")),
    ),
    "bnb_nf4": RecipeSpec(
        name="bnb_nf4",
        bits=4,
        method="bnb",
        extras=(("load_in_4bit", "True"), ("bnb_4bit_quant_type", "nf4")),
    ),
    "gptq_4bit": RecipeSpec(name="gptq_4bit", bits=4, method="gptq"),
    "awq_4bit": RecipeSpec(name="awq_4bit", bits=4, method="awq"),
}


def resolve(name: str) -> RecipeSpec:
    if name not in RECIPES:
        raise ValueError(f"unknown recipe '{name}'. Pick from {sorted(RECIPES)}")
    return RECIPES[name]


def known() -> list[str]:
    return sorted(RECIPES)
