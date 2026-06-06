from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from tabulate import tabulate

from ..recipes.registry import known
from ..runner import bench
from ..viz.charts import (
    plot_bits_vs_perplexity,
    plot_latency_bars,
    plot_perplexity_retention,
    plot_radar,
    plot_size_vs_perplexity,
)

app = typer.Typer(add_completion=False, help="qs: LLM quantization suite")


@app.command("bench")
def cmd_bench(
    model: Annotated[str, typer.Option(help="HF model id")] = "Qwen/Qwen2.5-0.5B-Instruct",
    recipes: Annotated[str, typer.Option(help="comma-separated recipe names")] = "fp16",
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
) -> None:
    recipe_list = [r.strip() for r in recipes.split(",") if r.strip()]
    for r in recipe_list:
        if r not in known():
            raise typer.BadParameter(f"unknown recipe {r!r}; choose from {known()}")
    results = bench(model, recipe_list, out_dir)
    rows = [
        (
            r.recipe,
            r.bits,
            f"{r.perplexity:.2f}",
            f"{r.model_size_mb:.1f}",
            f"{r.peak_mem_mb:.1f}",
            f"{r.load_secs:.1f}",
            f"{r.inference_ms_per_token:.1f}",
        )
        for r in results
    ]
    print()
    print(
        tabulate(
            rows,
            headers=["recipe", "bits", "PPL", "size MB", "peak MB", "load s", "ms/tok"],
            tablefmt="github",
        )
    )


@app.command("plots")
def cmd_plots(
    results_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
    figures_dir: Annotated[Path, typer.Option(help="figures dir")] = Path("results/figures"),
) -> None:
    plot_bits_vs_perplexity(results_dir, figures_dir / "bits_vs_perplexity.png")
    plot_size_vs_perplexity(results_dir, figures_dir / "size_vs_perplexity.png")
    plot_latency_bars(results_dir, figures_dir / "latency_bars.png")
    plot_perplexity_retention(results_dir, figures_dir / "perplexity_retention.png")
    plot_radar(results_dir, figures_dir / "recipe_radar.png")
    typer.echo(f"wrote 5 figures to {figures_dir}")


@app.command("recipes")
def cmd_recipes() -> None:
    for r in known():
        typer.echo(r)


if __name__ == "__main__":
    logger.info("qs CLI ready")
    app()
