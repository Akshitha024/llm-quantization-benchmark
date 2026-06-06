.PHONY: help install lint typecheck test bench plots clean

MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
RECIPE ?= fp16,bnb_8bit,bnb_4bit

help:
	@echo "make install                              - deps"
	@echo "make lint / typecheck / test              - quality gates"
	@echo "make bench MODEL=... RECIPE=fp16,...      - run perplexity + size + latency"
	@echo "  RECIPE choices: fp16 fp32 bnb_8bit bnb_4bit bnb_nf4 gptq_4bit awq_4bit"
	@echo "make plots                                - regenerate the 5 chart types"

install: ; uv sync --all-extras
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
typecheck: ; uv run mypy src
test: ; uv run pytest -m "not slow and not needs_gpu"
bench: ; uv run qs bench --model $(MODEL) --recipes $(RECIPE)
plots: ; uv run qs plots
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +


.PHONY: pdf test-artifacts
pdf:
	cd docs/_report && pandoc research_report.md -o ../research_report.pdf --pdf-engine=xelatex --toc --toc-depth=2 --number-sections -V geometry:margin=1in -V fontsize=11pt -V mainfont="Helvetica" -V monofont="Menlo" -V linkcolor=blue -V urlcolor=blue -V linestretch=1.15 || echo "pandoc + xelatex required; see https://pandoc.org/installing.html"

test-artifacts:
	uv run python ../../_meta/retrofit.py "$(notdir $(CURDIR))" "$(notdir $(CURDIR))"
