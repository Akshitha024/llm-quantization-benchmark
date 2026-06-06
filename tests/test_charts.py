from __future__ import annotations

import json
from pathlib import Path

from qs.viz.charts import (
    plot_bits_vs_perplexity,
    plot_latency_bars,
    plot_perplexity_retention,
    plot_radar,
    plot_size_vs_perplexity,
)


def _write_fake(
    rd: Path, recipe: str, bits: int, ppl: float, size: float, lat: float = 10.0, mem: float = 100.0
) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"m__{recipe}.json").write_text(
        json.dumps(
            {
                "model": "m",
                "recipe": recipe,
                "bits": bits,
                "perplexity": ppl,
                "model_size_mb": size,
                "peak_mem_mb": mem,
                "load_secs": 1.0,
                "inference_ms_per_token": lat,
                "extras": {},
            }
        )
    )


def test_bits_vs_perplexity_writes(tmp_path: Path) -> None:
    rd = tmp_path / "r"
    _write_fake(rd, "fp16", 16, 10.0, 500.0)
    _write_fake(rd, "bnb_4bit", 4, 11.5, 130.0)
    out = tmp_path / "f.png"
    plot_bits_vs_perplexity(rd, out)
    assert out.exists() and out.stat().st_size > 0


def test_size_vs_perplexity_writes(tmp_path: Path) -> None:
    rd = tmp_path / "r"
    _write_fake(rd, "fp16", 16, 10.0, 500.0)
    _write_fake(rd, "bnb_4bit", 4, 11.5, 130.0)
    _write_fake(rd, "bnb_8bit", 8, 10.2, 250.0)
    out = tmp_path / "f.png"
    plot_size_vs_perplexity(rd, out)
    assert out.exists() and out.stat().st_size > 0


def test_latency_bars_writes(tmp_path: Path) -> None:
    rd = tmp_path / "r"
    _write_fake(rd, "fp16", 16, 10.0, 500.0, lat=12.0)
    _write_fake(rd, "bnb_4bit", 4, 11.5, 130.0, lat=18.0)
    out = tmp_path / "f.png"
    plot_latency_bars(rd, out)
    assert out.exists() and out.stat().st_size > 0


def test_perplexity_retention_with_fp16_baseline(tmp_path: Path) -> None:
    rd = tmp_path / "r"
    _write_fake(rd, "fp16", 16, 10.0, 500.0)
    _write_fake(rd, "bnb_4bit", 4, 11.0, 130.0)
    out = tmp_path / "f.png"
    plot_perplexity_retention(rd, out)
    assert out.exists() and out.stat().st_size > 0


def test_radar_writes(tmp_path: Path) -> None:
    rd = tmp_path / "r"
    _write_fake(rd, "fp16", 16, 10.0, 500.0, lat=12.0, mem=200.0)
    _write_fake(rd, "bnb_4bit", 4, 11.5, 130.0, lat=18.0, mem=80.0)
    out = tmp_path / "f.png"
    plot_radar(rd, out)
    assert out.exists() and out.stat().st_size > 0


def test_handles_empty_results_gracefully(tmp_path: Path) -> None:
    rd = tmp_path / "r"
    rd.mkdir()
    out = tmp_path / "f.png"
    plot_bits_vs_perplexity(rd, out)
    assert out.exists()
