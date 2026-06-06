"""Sliding-window perplexity on a small text corpus.

Standard perplexity = exp(NLL / token_count). We use the windowed variant
because long inputs cannot be processed in one shot for many architectures;
this matches the HuggingFace evaluate-perplexity recipe.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from loguru import logger


@torch.no_grad()
def perplexity(
    model: Any,
    tokenizer: Any,
    text: str,
    max_length: int = 512,
    stride: int = 256,
) -> float:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    if seq_len < 4:
        logger.warning("perplexity input too short; returning inf")
        return float("inf")

    device = next(model.parameters()).device
    nlls: list[torch.Tensor] = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        ids = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore prefix

        out = model(ids, labels=target_ids)
        neg_log_likelihood = out.loss

        nlls.append(neg_log_likelihood.float() * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum() / max(1, prev_end_loc)
    return float(math.exp(total_nll.item()))


def perplexity_on_wikitext(model: Any, tokenizer: Any, n_chars: int = 50_000) -> float:
    """Convenience: pull a wikitext-2 snippet and measure perplexity."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(r["text"] for r in ds if r["text"].strip())[:n_chars]
    except Exception as e:
        logger.warning("wikitext load failed ({}); using a literary fallback", e)
        text = (
            "Call me Ishmael. Some years ago, never mind how long precisely, "
            "having little or no money in my purse, and nothing particular to "
            "interest me on shore, I thought I would sail about a little and "
            "see the watery part of the world. "
        ) * 100
    return perplexity(model, tokenizer, text)
