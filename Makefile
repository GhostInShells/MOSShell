.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "MOSS"
	@echo "  make install   — uv sync --all-extras"
	@echo "  make test      — pytest"
	@echo "  make lint      — ruff check"

.PHONY: install
install:
	uv sync --all-extras

.PHONY: test
test:
	uv run pytest

.PHONY: lint
lint:
	uv run ruff check
