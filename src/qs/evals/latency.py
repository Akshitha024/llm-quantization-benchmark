"""Per-token decode latency.

We warm up first (the first call after load is dominated by kernel compile
and tokenizer init), then time generate() over a fixed prompt and divide
by new tokens. Reports milliseconds per token.
"""

from __future__ import annotations

import time
from typing import Any

import torch


@torch.no_grad()
def decode_latency_ms_per_token(
    model: Any,
    tokenizer: Any,
    prompt: str = "The capital of France is",
    new_tokens: int = 64,
    warmup: int = 3,
) -> float:
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    for _ in range(warmup):
        model.generate(ids, max_new_tokens=8, do_sample=False)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    model.generate(
        ids,
        max_new_tokens=new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return (elapsed * 1000.0) / new_tokens
