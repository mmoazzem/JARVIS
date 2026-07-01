# Jarvis — housekeeping targets
#
# `make clean`      → remove Python caches + build cruft (safe, always)
# `make clean-logs` → remove daily log files in logs/ (keeps the dir)
# `make clean-all`  → clean + clean-logs
#
# config.yaml is intentionally NOT touched here — it's your install's state.
# Use `make clean-config` explicitly if you really want a fresh first-run.

.PHONY: clean clean-logs clean-all clean-config

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	@echo "cleaned: python caches + tool caches"

clean-logs:
	@rm -f logs/*.log
	@echo "cleaned: logs/*.log"

clean-all: clean clean-logs
	@echo "cleaned: all (caches + logs)"

# Deliberately separate and explicit — wiping config triggers the wizard next boot.
clean-config:
	@rm -f config/config.yaml
	@echo "removed: config/config.yaml (wizard will run on next boot)"