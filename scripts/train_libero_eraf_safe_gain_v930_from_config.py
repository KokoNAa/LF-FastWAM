#!/usr/bin/env python3
"""Reuse the actual resolved V9.29 data config; warm-start its V9.28 source."""
import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf


def launch_spec(config_path, gpus, environ, *, source_objective=29):
    if gpus < 1:
        raise ValueError("GPU count must be positive.")
    cfg = OmegaConf.load(config_path)
    eraf = cfg.model.policy_guard.entity_relation_grounding
    if eraf.grounding_objective_version != source_objective or not eraf.safe_gain_training:
        raise ValueError(
            f"Source config must be the completed V9.{source_objective} training run's config.yaml."
        )
    data = cfg.data.train
    native = list(data.dataset_dirs)
    cf = list(data.pgc_counterfactual_dataset_dirs)
    corrective = list(data.pgc_closed_loop_corrective_dataset_dirs)
    sidecars = list(data.pgc_entity_relation_sidecar_dirs)
    if (len(native), len(cf), len(corrective), len(sidecars)) != (2, 2, 1, 5):
        raise ValueError(
            "Expected verified V9.29 native/CF/corrective/sidecar counts 2/2/1/5."
        )
    first = Path(native[0]).resolve()
    suffix = "_no_noops_lerobot"
    if not first.name.endswith(suffix):
        raise ValueError(f"Unexpected native dataset: {first}")
    suite = first.name.removesuffix(suffix)
    if not cfg.get("resume"):
        raise ValueError("V9.29 config does not record its V9.28 warm checkpoint.")
    warm = Path(cfg.resume).resolve()
    # The shared launcher validates the checkpoint objective, step, full sidecar
    # bindings, action convention and workspace before touching CUDA.
    env = dict(environ)
    env.update(
        {
            "PYTHON_BIN": env.get("PYTHON_BIN", os.sys.executable),
            "LIBERO_DATA_ROOT": str(first.parent),
            "STATS_PATH": str(Path(data.pretrained_norm_stats).resolve()),
            "TEXT_CACHE_DIR": str(Path(data.text_embedding_cache_dir).resolve()),
        }
    )
    command = [
        "bash",
        "scripts/train_libero_eraf_safe_gain_v930.sh",
        suite,
        str(gpus),
        str(warm),
        *cf,
        *corrective,
        sidecars[0],
        native[1],
        sidecars[1],
        *sidecars[2:],
        str(cfg.seed),
    ]
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="actual V9.29 run-root/config.yaml (not .hydra)"
    )
    parser.add_argument("gpus", type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    os.chdir(repo)
    command, env = launch_spec(config, args.gpus, os.environ)
    print(f"source_config={config}\nv928_teacher={command[4]}", flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
