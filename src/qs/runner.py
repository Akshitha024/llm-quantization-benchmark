"""Orchestrator: load model under each recipe -> bench -> write JSON."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from loguru import logger

from .evals.latency import decode_latency_ms_per_token
from .evals.perplexity import perplexity_on_wikitext
from .recipes.loader import load
from .recipes.registry import resolve
from .types import BenchResult


def bench_one(model_id: str, recipe_name: str) -> BenchResult:
    spec = resolve(recipe_name)
    loaded = load(model_id, spec)
    try:
        ppl = perplexity_on_wikitext(loaded.model, loaded.tokenizer)
        lat = decode_latency_ms_per_token(loaded.model, loaded.tokenizer)
    finally:
        peak_mem = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        )
    return BenchResult(
        model=model_id,
        recipe=recipe_name,
        bits=spec.bits,
        perplexity=ppl,
        model_size_mb=loaded.bytes_on_device / (1024 * 1024),
        peak_mem_mb=peak_mem,
        load_secs=loaded.load_secs,
        inference_ms_per_token=lat,
    )


def bench(model_id: str, recipes: list[str], out_dir: Path) -> list[BenchResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[BenchResult] = []
    for r in recipes:
        try:
            res = bench_one(model_id, r)
        except Exception as e:
            logger.warning("recipe {} failed: {} (skipping)", r, e)
            continue
        results.append(res)
        path = out_dir / f"{_safe(model_id)}__{r}.json"
        path.write_text(json.dumps(_to_dict(res), indent=2))
        logger.info("wrote {}", path)
        # tear down GPU memory between recipes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return results


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _to_dict(r: BenchResult) -> dict[str, object]:
    return {
        "model": r.model,
        "recipe": r.recipe,
        "bits": r.bits,
        "perplexity": r.perplexity,
        "model_size_mb": r.model_size_mb,
        "peak_mem_mb": r.peak_mem_mb,
        "load_secs": r.load_secs,
        "inference_ms_per_token": r.inference_ms_per_token,
        "extras": r.extras,
    }
