import os
import yaml
import time
import random
import numpy as np
import pandas as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.model_selection import train_test_split

# --- Comet ML (optional) ---
try:
    from comet_ml import Experiment  # type: ignore
except Exception:
    Experiment = None


def init_comet(args: dict):
    """
    Initialize Comet experiment if comet config is present and comet_ml is installed.
    Expects args to be a dict (loaded from YAML).
    """
    comet_cfg = args.get("comet", {}) if isinstance(args.get("comet", {}), dict) else {}
    if not comet_cfg or Experiment is None:
        return None

    api_key = comet_cfg.get("api_key") or os.environ.get("COMET_API_KEY")
    project_name = comet_cfg.get("project_name") or os.environ.get("COMET_PROJECT_NAME")
    workspace = comet_cfg.get("workspace") or os.environ.get("COMET_WORKSPACE")

    if not api_key or not project_name:
        return None

    exp = Experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspace,
        auto_param_logging=False,
        auto_metric_logging=False,
        auto_output_logging="simple",
    )

    # Log the args (minus nested comet dict)
    experiment = exp
    experiment.log_parameters({k: v for k, v in args.items() if k != "comet"})
    return experiment


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args():
    parser = ArgumentParser(
        description="ECE600 Credit Card Fraud Detection (GTAN / RGTAN runner)",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", type=str, required=True, help="gtan or rgtan")
    parser.add_argument("--dataset", type=str, required=False, default=None, help="sffsd or ieee")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return vars(parser.parse_args())


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def merge_args_config(cli_args: dict, cfg: dict) -> dict:
    """
    CLI args take precedence where provided; config provides defaults.
    """
    merged = dict(cfg)
    for k, v in cli_args.items():
        if v is not None:
            merged[k] = v
    return merged


def main(cli_args: dict):
    cfg = load_yaml_config(cli_args["config"])
    args = merge_args_config(cli_args, cfg)

    # Seed
    seed = int(args.get("seed", 42))
    seed_everything(seed)

    # Optional Comet
    experiment = init_comet(args)

    method = (args.get("method") or "").lower()
    dataset = (args.get("dataset") or "").lower() if args.get("dataset") else None

    # Import methods lazily to avoid heavy imports when not needed
    if method == "gtan":
        from methods.gtan.gtan_main import load_gtan_data, train_gtan  # type: ignore

        # Dataset routing
        if dataset == "sffsd":
            # Example: expects args to specify data path and columns consistent with S-FFSD
            # Use your existing load function logic (gtan_main.py) as source of truth.
            print("[INFO] Loading S-FFSD dataset for GTAN...")
            feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(args)
            print("[INFO] Training GTAN...")
            print(f"[RUN] method={args['method']} dataset={args.get('dataset')}")
            train_gtan(args, feat_data, labels, train_idx, test_idx, g, cat_features, experiment)

        elif dataset == "ieee":
            print("[INFO] Loading IEEE dataset for GTAN...")
            feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(args)
            print("[INFO] Training GTAN...")
            print(f"[RUN] method={args['method']} dataset={args.get('dataset')}")
            train_gtan(args, feat_data, labels, train_idx, test_idx, g, cat_features, experiment)

        else:
            # If dataset not specified, attempt default path in args
            print("[WARN] No dataset specified; proceeding with load_gtan_data(args) using config paths.")
            feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(args)
            print("[INFO] Training GTAN...")
            train_gtan(args, feat_data, labels, train_idx, test_idx, g, cat_features, experiment)

    elif method == "rgtan":
        from methods.rgtan.rgtan_main import load_rgtan_data, train_rgtan  # type: ignore

        if dataset == "sffsd":
            print("[INFO] Loading S-FFSD dataset for RGTAN...")
            feat_data, labels, train_idx, test_idx, g, cat_features = load_rgtan_data(args)
            print("[INFO] Training RGTAN...")
            print(f"[RUN] method={args['method']} dataset={args.get('dataset')}")
            train_rgtan(args, feat_data, labels, train_idx, test_idx, g, cat_features, experiment)

        elif dataset == "ieee":
            print("[INFO] Loading IEEE dataset for RGTAN...")
            feat_data, labels, train_idx, test_idx, g, cat_features = load_rgtan_data(args)
            print("[INFO] Training RGTAN...")
            print(f"[RUN] method={args['method']} dataset={args.get('dataset')}")
            train_rgtan(args, feat_data, labels, train_idx, test_idx, g, cat_features, experiment)

        else:
            print("[WARN] No dataset specified; proceeding with load_rgtan_data(args) using config paths.")
            feat_data, labels, train_idx, test_idx, g, cat_features = load_rgtan_data(args)
            print("[INFO] Training RGTAN...")
            train_rgtan(args, feat_data, labels, train_idx, test_idx, g, cat_features, experiment)

    else:
        raise ValueError(f"Unknown method: {method}. Expected 'gtan' or 'rgtan'.")

    if experiment is not None:
        try:
            experiment.end()
        except Exception:
            pass


if __name__ == "__main__":
    main(parse_args())
