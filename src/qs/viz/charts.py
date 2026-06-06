"""Five charts for the quantization sweep.

Distinct from prior projects:
  - bits vs perplexity (line/scatter, log y)
  - size vs perplexity Pareto frontier (scatter, annotated)
  - per-recipe latency bar (horizontal)
  - perplexity-retention vs fp16 baseline (percent bars)
  - radar/spider of normalized metrics per recipe
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _load(results_dir: Path) -> list[dict[str, Any]]:
    out = []
    for f in sorted(results_dir.glob("*.json")):
        if f.stem in {"summary"}:
            continue
        try:
            o = json.loads(f.read_text())
            if "perplexity" in o and "recipe" in o:
                out.append(o)
        except (json.JSONDecodeError, OSError):
            continue
    return out


# 1. Bits vs perplexity
def plot_bits_vs_perplexity(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load(results_dir)
    if not rows:
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(r["bits"], r["perplexity"], s=140, alpha=0.7, edgecolor="black")
        ax.annotate(
            r["recipe"],
            (r["bits"], r["perplexity"]),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("perplexity (wikitext-2)")
    ax.set_yscale("log")
    ax.set_xticks([4, 8, 16, 32])
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Bits per weight vs perplexity (log y)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 2. Size vs perplexity Pareto
def plot_size_vs_perplexity(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load(results_dir)
    if not rows:
        out.write_bytes(b"")
        return out
    sizes = [r["model_size_mb"] for r in rows]
    ppls = [r["perplexity"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sizes, ppls, s=140, c=ppls, cmap="viridis_r", edgecolor="black")
    for r, s, p in zip(rows, sizes, ppls, strict=True):
        ax.annotate(r["recipe"], (s, p), textcoords="offset points", xytext=(7, 6), fontsize=9)
    # naive Pareto frontier: sort by size, keep min-perplexity-so-far
    order = sorted(zip(sizes, ppls, strict=True), key=lambda x: x[0])
    front_x, front_y = [], []
    best = float("inf")
    for x, y in order:
        if y < best:
            best = y
            front_x.append(x)
            front_y.append(y)
    ax.plot(front_x, front_y, "k--", alpha=0.5, label="Pareto frontier")
    ax.set_xlabel("model size on device (MB)")
    ax.set_ylabel("perplexity")
    ax.set_title("Size vs perplexity Pareto")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 3. Per-recipe latency horizontal bar
def plot_latency_bars(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load(results_dir)
    if not rows:
        out.write_bytes(b"")
        return out
    rows = sorted(rows, key=lambda r: r["inference_ms_per_token"])
    names = [r["recipe"] for r in rows]
    lats = [r["inference_ms_per_token"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(rows) + 1)))
    bars = ax.barh(names, lats, color="#1f77b4")
    for bar, v in zip(bars, lats, strict=True):
        ax.text(
            bar.get_width() + max(lats) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("ms per generated token")
    ax.set_title("Decode latency by recipe")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 4. Perplexity retention (relative to fp16 baseline) as percent bars
def plot_perplexity_retention(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load(results_dir)
    if not rows:
        out.write_bytes(b"")
        return out
    baseline = next((r for r in rows if r["recipe"] == "fp16"), None)
    if baseline is None:
        baseline = next((r for r in rows if r["bits"] >= 16), None)
    if baseline is None:
        out.write_bytes(b"")
        return out
    base_ppl = baseline["perplexity"]
    rows = [r for r in rows if r["recipe"] != baseline["recipe"]]
    names = [r["recipe"] for r in rows]
    # retention = base_ppl / candidate_ppl: 1.0 means identical, <1 means worse
    retention = [base_ppl / r["perplexity"] for r in rows]
    colors = ["#2ca02c" if x >= 0.95 else "#ff7f0e" if x >= 0.85 else "#d62728" for x in retention]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(rows) + 1)))
    bars = ax.barh(names, retention, color=colors)
    ax.axvline(1.0, color="black", linewidth=1, linestyle=":", label="fp16 parity")
    for bar, v in zip(bars, retention, strict=True):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v * 100:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max(1.1, max(retention) * 1.05))
    ax.set_xlabel(f"perplexity retention vs {baseline['recipe']}")
    ax.set_title(f"Quality retention relative to {baseline['recipe']} (1.0 = no loss)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 5. Per-recipe normalized radar
def plot_radar(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load(results_dir)
    if not rows:
        out.write_bytes(b"")
        return out
    axes_keys = [
        "perplexity",
        "model_size_mb",
        "peak_mem_mb",
        "load_secs",
        "inference_ms_per_token",
    ]
    mats = []
    for r in rows:
        mats.append([float(r.get(k, 0)) for k in axes_keys])
    mat = np.array(mats, dtype=np.float64)
    # invert so "more is better" for the radar; perplexity, size, mem, load, latency all benefit from min
    inv = 1.0 / (mat + 1e-9)
    norm = inv / (inv.max(axis=0) + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(axes_keys), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for r, vals in zip(rows, norm, strict=True):
        values = [*vals.tolist(), vals[0]]
        ax.plot(angles, values, linewidth=2, marker="o", label=r["recipe"])
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["1/PPL", "1/size", "1/mem", "1/load", "1/latency"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.05), fontsize=8)
    ax.set_title("Normalized 'better-is-larger' radar across recipes", pad=20)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
