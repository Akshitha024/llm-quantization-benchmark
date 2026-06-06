"""Core types for quantization benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecipeSpec:
    name: str  # fp16, bnb_4bit, gptq_4bit, ...
    bits: int
    method: str  # "none", "bnb", "gptq", "awq"
    extras: tuple[tuple[str, str], ...] = ()  # config flags


@dataclass
class BenchResult:
    model: str
    recipe: str
    bits: int
    perplexity: float
    model_size_mb: float
    peak_mem_mb: float
    load_secs: float
    inference_ms_per_token: float
    extras: dict[str, float] = field(default_factory=dict)
