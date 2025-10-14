from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
import torch
import pandas as pd

from active_learning_loop import active_learning_loop


# ---------- Path helpers ----------

def project_root() -> Path:
    """
    Return the project root = parent of this file's directory.
    If this file is at <root>/src/run_experiments.py, then root is parent of src.
    """
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    """Default to <project_root>/config.yaml"""
    return project_root() / "config.yaml"


def resolve_config_path(user_path: Optional[str]) -> Path:
    """
    Resolve a config path with robust rules:

    - If user_path is None: use default_config_path()
    - If user_path is absolute: use it as-is
    - If user_path is relative: resolve relative to the *current working directory*
      (i.e., where the user ran the script), not relative to this file.
    """
    if user_path is None:
        path = default_config_path()
    else:
        p = Path(user_path).expanduser()
        path = p if p.is_absolute() else (Path.cwd() / p)

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


# ---------- Config loader ----------

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    path = resolve_config_path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- Experiment runners ----------

def run_single_experiment(experiment_config: dict, experiment_idx: Optional[int] = None) -> Path:
    """Run a single experiment with given configuration and return its results directory."""

    # Basic validation and defaults
    def _get(cfg, *keys, default=None):
        cur = cfg
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # validate presence of some important fields early
    if not isinstance(experiment_config, dict):
        raise ValueError("Experiment config must be a mapping/dict")
    if _get(experiment_config, "active_learning", "acquisition_functions") is None:
        raise ValueError("Each experiment must define active_learning.acquisition_functions")
    if _get(experiment_config, "output", "results_dir") is None:
        raise ValueError("Each experiment must define output.results_dir")

    # Device setup
    device = experiment_config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = experiment_config.get("name", "experiment")
    exp_label = f"Experiment {experiment_idx}: {exp_name}" if experiment_idx is not None else exp_name

    print(f"\n{'=' * 80}")
    print(exp_label)
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    print(f"Seed: {experiment_config.get('seed')}")
    acq_list = _get(experiment_config, "active_learning", "acquisition_functions", default=[])
    print(f"Acquisition functions: {acq_list}")
    print(f"{'=' * 80}\n")

    # Create output directory with timestamp under project root (unless user gave an absolute path)
    root = project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve results_dir: allow absolute or relative (relative = under project root)
    configured_results_dir = experiment_config["output"]["results_dir"]
    configured_results_dir = Path(configured_results_dir)
    if not configured_results_dir.is_absolute():
        configured_results_dir = root / configured_results_dir

    results_dir = configured_results_dir / f"{exp_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}\n")

    # Run experiments for each acquisition function
    for acq in experiment_config["active_learning"]["acquisition_functions"]:
        print(f"\n{'-' * 60}")
        print(f"Running acquisition function: {acq}")
        print(f"{'-' * 60}\n")

        try:
            # Build kwargs for active_learning_loop from config, providing sensible defaults when missing
            al_cfg = experiment_config.get("active_learning", {})
            data_cfg = experiment_config.get("data", {})
            training_cfg = experiment_config.get("training", {})
            output_cfg = experiment_config.get("output", {})

            df = active_learning_loop(
                BASE_DIR=experiment_config.get("base_dir"),
                LABEL_SPLIT_RATIO=data_cfg.get("label_split_ratio", 0.1),
                TEST_SPLIT_RATIO=data_cfg.get("test_split_ratio", 0.2),
                augment=data_cfg.get("augment", False),
                sample_size=al_cfg.get("sample_size", 2),
                acquisition_type=acq,
                mc_runs=al_cfg.get("mc_runs", 8),
                dropout=al_cfg.get("dropout", 0.3),
                batch_size=training_cfg.get("batch_size", 16),
                lr=training_cfg.get("lr", 1e-3),
                seed=experiment_config.get("seed"),
                loop_iterations=training_cfg.get("loop_iterations", None),
                device=device,
                patience=training_cfg.get("patience", 15),
                min_delta=training_cfg.get("min_delta", 1e-4),
                ssl_lambda_max=al_cfg.get("ssl_lambda_max", 1.0),
                ssl_ramp_epochs=al_cfg.get("ssl_ramp_epochs", 10),
            )

            # Add metadata columns
            if isinstance(df, pd.DataFrame):
                df["method"] = acq
                df["experiment"] = exp_name

                # Save individual result
                log_fmt = output_cfg.get("log_format", "{method}.csv")
                output_path = results_dir / log_fmt.format(method=acq)
                df.to_csv(output_path, index=False)
                print(f"✓ Saved results to {output_path}")
            else:
                print("Warning: active_learning_loop did not return a pandas DataFrame; skipping save.")

        except Exception as e:
            print(f"✗ Error running {acq}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save experiment config for reproducibility
    config_copy_path = results_dir / "config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(experiment_config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✓ Experiment configuration saved to {config_copy_path}")
    return results_dir


def run_experiments(config_path: Optional[str] = None) -> None:
    """Run all experiments from a config file (supports both single and multiple experiment formats)."""
    config = load_config(config_path)

    # New format: a list of experiments under 'experiments'
    if "experiments" in config and isinstance(config["experiments"], list):
        experiments = config["experiments"]
        print(f"\nFound {len(experiments)} experiment(s) to run\n")

        results_dirs: list[Path] = []
        for idx, exp_config in enumerate(experiments, 1):
            try:
                results_dir = run_single_experiment(exp_config, idx)
                results_dirs.append(results_dir)
            except Exception as e:
                print(f"\n✗ Failed to complete experiment {idx}: {str(e)}\n")
                continue

        print(f"\n{'=' * 80}")
        print("All experiments completed!")
        print(f"Results saved in {len(results_dirs)} directorie(s)")
        print(f"{'=' * 80}\n")

    else:
        # Old format: single experiment dict
        print("\nRunning single experiment (legacy config format)\n")
        run_single_experiment(config)
        print(f"\n{'=' * 80}")
        print("Experiment completed!")
        print(f"{'=' * 80}\n")


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run active learning experiments")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file. If omitted, uses <project_root>/config.yaml",
    )
    args = parser.parse_args()

    resolved = resolve_config_path(args.config) if args.config is not None else default_config_path()
    print(f"Using config: {resolved}")
    run_experiments(str(resolved))
