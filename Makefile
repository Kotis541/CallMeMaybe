UV_CACHE = /sgoinfre/$(USER)/.uv_cache

.PHONY: install run debug clean lint

install:
	UV_CACHE_DIR=$(UV_CACHE) uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src

clean:
	rm -rf src/__pycache__ llm_sdk/llm_sdk/__pycache__* __pycache__ .mypy_cache .pytest_cache
	rm -rf $(UV_CACHE)

lint:
	flake8  src/* test_statemachine.py
	mypy src/* --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs