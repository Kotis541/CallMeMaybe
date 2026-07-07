.PHONY: install run debug clean lint

install:
	uv sync

run:
	python main.py

debug:
	python -m pdb main.py

clean:
	rm -rf src/__pycache__ llm_sdk/llm_sdk/__pycache__*

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs