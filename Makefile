export UV_CACHE_DIR = /tmp/$(USER)_uv_cache
export HF_HOME      = /sgoinfre/$(USER)/.cache/huggingface


.PHONY: install run debug clean lint test

install:
	uv sync

run:
	uv run python -m src 

debug:
	uv run python -m pdb src $(ARGS)

clean:
	rm -rf src/__pycache__ llm_sdk/llm_sdk/__pycache__* __pycache__ .mypy_cache .pytest_cache
	rm -rf $(UV_CACHE)
	rm -rf $(HF_HOME)

lint:
	flake8  src/* test_statemachine.py
	mypy src/* --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	uv run python -m pytest