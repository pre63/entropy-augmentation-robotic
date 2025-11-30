SHELL := /bin/bash

OS := $(shell uname -s)

DEVICE=cpu


ubuntu:
	@if [ "$(OS)" != "Linux" ]; then \
		YPATH=".noise/2025-03-10_07-15-27/GenPPO_100000_reward_action_5_runs.yml;.noise/2025-03-09_07-39-20/GenPPO_100000_reward_action_5_runs.yml;.noise/2025-03-08_10-51-44/GenPPO_100000_reward_action_5_runs.yml;.noise/2025-03-05_18-34-02/GenTRPO_100000_reward_action_5_runs.yml;.noise/2025-03-05_02-24-05/PPO_100000_reward_action_5_runs.yml;.noise/2025-03-05_02-24-05/TRPO_100000_reward_action_5_runs.yml;.noise/2025-03-05_02-24-05/TRPOER_100000_reward_action_5_runs.yml;.noise/2025-03-05_02-24-05/TRPOR_100000_reward_action_5_runs.yml"
	elif ! command -v lsb_release > /dev/null; then \
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

install: ubuntu venv

fix:
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

experiments: fix
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.experiments

report: fix
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.report
