import os
import yaml
import time
import random
import numpy as np
import pandas as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from comet_ml import Experiment


def init_comet(args: dict):
    """
    Initializes Comet ML experiment if api_key is present in YAML config.
    """
    comet_cfg = args.get("comet", {}) if isinstance(args.get("comet", {}), dict) else {}
    api_key = comet_cfg.get("api_key") or os.environ.get("COMET_API_KEY")

    if not api_key:
        return None

    experiment = Experiment(
        api_key=api_key,
        project_name=comet_cfg.get("project_name", "antifraud"),
        workspace=comet_cfg.get("workspace", None),
        auto_metric_logging=False,
        auto_param_logging=False,
        auto_output_logging=False,
    )

    experiment.log_parameters({k: v for k, v in args.items() if k != "comet"})
    return experiment


def log_comet_status(experiment):
    if experiment is None:
        print("[COMET] Not enabled (no API key).", flush=True)
    else:
        print("[COMET] Enabled.", flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args():
    """
    Loads the method's YAML config, then applies optional CLI overrides:
      python main.py --method gtan --dataset IEEE --ieee-mode v2
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )

    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", default=None)

    parser.add_argument(
        "--ieee-mode",
        default="auto",
        choices=["auto", "raw", "norm", "v2"],
        help=(
            "IEEE loader mode:\n"
            "  raw  = original train_transaction.csv\n"
            "  norm = S-FFSD-like normalized file\n"
            "  v2   = entity-gated v2 graph\n"
            "  auto = prefer norm if present, else raw"
        ),
    )

    args_cli = vars(parser.parse_args())
    method = args_cli["method"]

    if method == "mcnn":
        yaml_file = "config/mcnn_cfg.yaml"
    elif method == "stan":
        yaml_file = "config/stan_cfg.yaml"
    elif method == "stan_2d":
        yaml_file = "config/stan_2d_cfg.yaml"
    elif method == "gtan":
        yaml_file = "config/gtan_cfg.yaml"
    elif method == "rgtan":
        yaml_file = "config/rgtan_cfg.yaml"
    elif method == "hogrl":
        yaml_file = "config/hogrl_cfg.yaml"
    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    with open(yaml_file, "r") as f:
        args = yaml.safe_load(f)

    args["method"] = method

    if args_cli.get("dataset"):
        args["dataset"] = args_cli["dataset"]

    args["ieee_mode"] = args_cli.get("ieee_mode", "auto")

    return args


def span_data_3d(feat_df):
    """
    Placeholder for stan path; not modified here.
    """
    raise NotImplementedError


def span_data_2d(feat_df):
    """
    Placeholder for stan_2d path; not modified here.
    """
    raise NotImplementedError


def base_load_data(args: dict):
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args["test_size"]

    if args["method"] == "stan":
        features, labels = span_data_3d(feat_df)
    else:
        features, labels = span_data_2d(feat_df)

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        train_size=train_size,
        stratify=labels,
        random_state=args["seed"],
    )

    trf = features[train_idx]
    tef = features[test_idx]
    trl = labels[train_idx]
    tel = labels[test_idx]

    np.save(args["trainfeature"], trf)
    np.save(args["testfeature"], tef)
    np.save(args["trainlabel"], trl)
    np.save(args["testlabel"], tel)


def main(args):
    experiment = init_comet(args)
    log_comet_status(experiment)

    try:
        if args["method"] in {"mcnn", "stan", "stan_2d"}:
            base_load_data(args)

        if args["method"] == "gtan":
            from methods.gtan.gtan_main import gtan_main, load_gtan_data

            # Support both signatures:
            #   load_gtan_data(dataset=..., ...)
            # and
            #   load_gtan_data(prefix=..., dataset=..., ...)
            import inspect

            sig = inspect.signature(load_gtan_data)
            if "prefix" in sig.parameters:
                prefix = args.get("prefix") or args.get("data_dir") or args.get("data_path") or "data"
                feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
                    prefix=prefix,
                    dataset=args["dataset"],
                    test_size=args["test_size"],
                    ieee_mode=args["ieee_mode"],
                )
            else:
                feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
                    dataset=args["dataset"],
                    test_size=args["test_size"],
                    ieee_mode=args["ieee_mode"],
                )

            gtan_main(
                feat_data,
                g,
                train_idx,
                test_idx,
                labels,
                args,
                cat_features,
                experiment=experiment,
            )

        if args["method"] == "rgtan":
            from methods.rgtan.rgtan_main import rgtan_main, loda_rgtan_data

            feat_data, labels, train_idx, test_idx, g, cat_features = loda_rgtan_data(
                dataset=args["dataset"],
                test_size=args["test_size"],
            )

            rgtan_main(
                feat_data,
                g,
                train_idx,
                test_idx,
                labels,
                args,
                cat_features,
                experiment=experiment,
            )

    finally:
        if experiment is not None:
            try:
                experiment.end()
            except Exception:
                pass


if __name__ == "__main__":
    main(parse_args())
