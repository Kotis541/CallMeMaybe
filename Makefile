UV_CACHE = /sgoinfre/$(USER)/.uv_cache
HF_CACHE = /sgoinfre/$(USER)/.hf_cache

export UV_CACHE_DIR = $(UV_CACHE)
export UV_PROJECT_ENVIRONMENT = $(UV_VENV)
export HF_HOME = $(HF_CACHE)

.PHONY: install run debug clean lint test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src

clean:
	rm -rf src/__pycache__ llm_sdk/llm_sdk/__pycache__* __pycache__ .mypy_cache .pytest_cache
	# Smaže cache i virtuální prostředí ve sgoinfre
	rm -rf $(UV_CACHE)
	rm -rf $(HF_CACHE)

lint:
	flake8 src/* test_statemachine.py
	mypy src/* --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	uv run python -m pytest