#!/usr/bin/env python3
"""Run V9.36 action-violation-gated semantic contrast."""

try:
    from scripts.train_libero_eraf_safe_gain_v932 import main
except ModuleNotFoundError:
    from train_libero_eraf_safe_gain_v932 import main


if __name__ == "__main__":
    main(objective=36)
