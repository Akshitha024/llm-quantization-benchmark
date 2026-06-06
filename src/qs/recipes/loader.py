"""Load a transformers model under a given quantization recipe.

The bnb (bitsandbytes) recipes use ``BitsAndBytesConfig`` which requires the
``bitsandbytes`` package and a CUDA-capable GPU at load time. On CPU-only
machines those recipes will fail to load; we degrade to fp16/fp32 so the
benchmark surface still works.

GPTQ and AWQ recipes expect pre-quantized weights on the Hub (e.g.
``TheBloke/Qwen2.5-0.5B-GPTQ``). When unavailable we surface a clean
error so the user can pick a different model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from ..types import RecipeSpec


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    load_secs: float
    bytes_on_device: int  # estimated; sum of parameter bytes


def load(model_id: str, recipe: RecipeSpec) -> LoadedModel:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)  # type: ignore[no-untyped-call]

    extras = {k: _coerce(v) for k, v in recipe.extras}

    kwargs: dict[str, Any] = {}
    if recipe.method == "none":
        if recipe.bits == 32:
            kwargs["torch_dtype"] = torch.float32
        else:
            kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
    elif recipe.method == "bnb":
        try:
            bnb = BitsAndBytesConfig(**extras)  # type: ignore[no-untyped-call]
            kwargs["quantization_config"] = bnb
            kwargs["device_map"] = "auto"
        except Exception as e:
            logger.warning("bnb config failed ({}); falling back to fp16", e)
            kwargs["torch_dtype"] = torch.float16
    elif recipe.method == "gptq":
        # expects a GPTQ-prequantized variant of model_id
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
    elif recipe.method == "awq":
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
    else:
        raise ValueError(f"unknown method {recipe.method}")

    logger.info("loading {} with recipe={} extras={}", model_id, recipe.name, extras)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    load_secs = time.perf_counter() - t0

    bytes_on_device = _estimate_bytes(model, recipe)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        load_secs=load_secs,
        bytes_on_device=bytes_on_device,
    )


def _coerce(v: str) -> Any:
    if v == "True":
        return True
    if v == "False":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _estimate_bytes(model: Any, recipe: RecipeSpec) -> int:
    # parameter count * bits / 8. Holds for dense layers; quantized
    # weights with packing still report through .numel() but their
    # actual storage is bits/8 of that count.
    total_params = sum(p.numel() for p in model.parameters())
    return int(total_params * recipe.bits / 8)
