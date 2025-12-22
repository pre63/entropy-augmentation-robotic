SHELL := /bin/bash

OS := $(shell uname -s)

DEVICE=cpu

default: experiments report

ubuntu:
	if ! command -v lsb_release > /dev/null; then \
		echo "lsb_release not found, skipping Ubuntu setup."; \
	elif ! lsb_release -a 2>/dev/null | grep -q "Ubuntu"; then \
		echo "Not an Ubuntu system, skipping."; \
	else \
		echo "Running Ubuntu setup..."; \
		sudo apt-get update && \
		sudo apt-get -y install python3-dev swig build-essential cmake && \
		sudo apt-get -y install python3.12-venv python3.12-dev && \
		sudo apt-get -y install swig python-box2d; \
	fi

venv:
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

install: venv

fix: setup
	@echo "Will run black and isort on modified, added, untracked, or staged Python files"
	@changed_files=$$(git diff --name-only --diff-filter=AM | grep '\.py$$'); \
	untracked_files=$$(git ls-files --others --exclude-standard | grep '\.py$$'); \
	staged_files=$$(git diff --name-only --cached | grep '\.py$$'); \
	all_files=$$(echo "$$changed_files $$untracked_files $$staged_files" | tr ' ' '\n' | sort -u); \
	if [ ! -z "$$all_files" ]; then \
		. .venv/bin/activate && isort --multi-line=0 --line-length=100 $$all_files && black .; \
	else \
		echo "No modified, added, untracked, or staged Python files"; \
	fi

clean:
	@echo "Cleaning up"
	@rm -rf __pycache__/
	@rm -rf .venv

setup: 
	@mkdir -p ~logs
	@mkdir -p reports
	
experiments: fix setup
	@source .venv/bin/activate; \
	CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.experiments 2>&1 | tee -a ~logs/experiments

report: fix setup
	@rm -f reports/.tmp
	@source .venv/bin/activate; \
	CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.report 2>&1 | tee -a reports/.tmp
	@mv reports/.tmp reports/report.md

actions: fix
	@source .venv/bin/activate; \
	CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.actions

tune: fix
	@mkdir -p ~logs
	@touch ~logs/tune
	@source .venv/bin/activate; \
	CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.tune 2>&1 | tee -a ~logs/tune

optuna: fix
	@source .venv/bin/activate; \
	CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.optuna 2>&1 | tee -a ~logs/optuna