.DEFAULT_GOAL := prepare

export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: help
help: ## Show available make targets.
	@echo "Available make targets:"
	@awk 'BEGIN { FS = ":.*## " } /^[A-Za-z0-9_.-]+:.*## / { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: install-uv
install-uv: ## Install uv if missing
	@echo "==> Checking for uv"
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv already installed at $$(command -v uv)"; \
	else \
		echo "uv not found. Installing via curl script..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

.PHONY: install-python
install-python: ## Install Python via uv if missing
	@echo "==> Ensuring Python 3.12 is available (via uv)"
	@if uv python find 3.12 >/dev/null 2>&1; then \
		echo "Python 3.12 already available"; \
	else \
		echo "Python 3.12 not found. Installing..."; \
		uv python install 3.12; \
	fi

.PHONY: uv-venv
uv-venv: ## Create project virtualenv with uv if missing
	@echo "==> Checking for .venv"
	@if [ -d .venv ]; then \
		echo ".venv already exists"; \
	else \
		echo "Creating .venv with uv"; \
		uv venv; \
	fi

.PHONY: install-uv-pyenv
install-uv-pyenv: install-uv install-python uv-venv ## Install uv, Python 3.12, and venv

.PHONY: install-prek
install-prek: ## Install prek and repo git hooks.
	@echo "==> Installing prek"
	@uv tool install prek
	@echo "==> Installing git hooks with prek"
	@uv tool run prek install

.PHONY: prepare
prepare: install-uv install-python uv-venv install-prek ## Setup uv, Python 3.12, venv, and prek hooks.
	@echo "==> Syncing dependencies for all workspace packages"
	@uv sync --dev --all-extras

MDFORMAT := $(shell if [ -x .venv/bin/mdformat ]; then echo .venv/bin/mdformat; else echo "uv run --dev mdformat"; fi)

.PHONY: format
format: ## Run format
	@echo "==> Formatting"
	@uv run --dev ruff format
	@git ls-files -z '*.md' | xargs -0 $(MDFORMAT)

.PHONY: lint
lint: ## Run lint
	@echo "==> Linting"
	@uv run --dev ruff check

.PHONY: type-check
type-check:
	@echo "==> checking types"
	@uv run --dev ty check
	# @uv run --dev basedpyright

.PHONY: test
test: ## Run pytest
	@echo "==> Testing"
	@uv run --dev pytest
