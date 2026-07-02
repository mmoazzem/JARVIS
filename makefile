# Jarvis — housekeeping targets
#
# `make clean`        → remove Python caches + build cruft (safe, always)
# `make clean-logs`   → remove daily log files in logs/ (keeps the dir)
# `make clean-config` → remove config.yaml (wizard runs on next boot)
# `make clean-all`    → EVERYTHING: caches + logs + config (full reset to first-run)

.PHONY: clean clean-logs clean-config clean-all

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

clean-config:
	@rm -f config/config.yaml
	@echo "removed: config/config.yaml (wizard will run on next boot)"

# Full reset: caches + logs + config. Next boot is a fresh first-run.
clean-all: clean clean-logs clean-config
	@echo "cleaned: ALL (caches + logs + config) — next boot is first-run"