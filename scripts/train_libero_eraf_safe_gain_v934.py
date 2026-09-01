#!/usr/bin/env python3
"""Run V9.34 universal positive-only paired ranking with the matched workflow."""

try:
    from scripts.train_libero_eraf_safe_gain_v932 import main
except ModuleNotFoundError:
    from train_libero_eraf_safe_gain_v932 import main


if __name__ == "__main__":
    main(objective=34)
