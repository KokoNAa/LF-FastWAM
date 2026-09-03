#!/usr/bin/env python3
"""Validate an executable RoboTwin same-scene CIS manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.language_interventions import load_intervention_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "RoboTwin",
    )
    args = parser.parse_args()
    pairs = load_intervention_manifest(
        args.manifest,
        robotwin_root=args.robotwin_root,
    )
    print(f"Validated {len(pairs)} executable RoboTwin CIS records:")
    for pair in pairs:
        print(f"  {pair.pair_id}: {pair.source_task} -> " f"{pair.counterfactual_task}")


if __name__ == "__main__":
    main()
