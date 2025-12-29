import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from config import Config
from feature_engineering.data_engineering import (
    data_engineer_benchmark,
    span_data_2d,
    span_data_3d,
)

logger = logging.getLogger(__name__)

# --- Optional Comet support ---
try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


def init_comet(args: dict):
    """
    Create ONE Comet experiment per run.
    """
    if Experiment is None:
        return None

    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        return None

    exp = Experiment(
        api_key=api_key,
        workspace=os.getenv("COMET_WORKSPACE"),
        project_name=os.getenv("COMET_PROJECT_NAME", "ccfraud-gnn"),
        auto_param_logging=False,
        auto_metric_logging=False,
        auto_output_logging="simple",
        disabled=(os.getenv("COMET_API_KEY") is None),
    )

    try:
        exp.set_name(f"{args.get('method')}-{args.get('dataset')}-seed{args.get('seed')}")
        exp.add_tag(str(args.get("method")))
        exp.add_tag(str(args.get("dataset")))
        exp.log_parameters(args)
    except Exception:
        pass

    return exp


def log_comet_status(experiment):
    print("----- COMET STATUS -----")
    api_key = os.getenv("COMET_API_KEY")
    print("COMET_API_KEY set:", "YES" if api_key else "NO")
    if api_key:
        print("COMET_API_KEY preview:", api_key[:6] + "..." + api_key[-4:])
    print("COMET_WORKSPACE:", os.getenv("COMET_WORKSPACE"))
    print("COMET_PROJECT_NAME:", os.getenv("COMET_PROJECT_NAME"))

    if experiment is None:
        print("Experiment object: None (Comet disabled)")
    else:
        print("Experiment object created:", True)
        print("Experiment disabled:", experiment.disabled)

    print("------------------------")


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
    elif method == "stagn":
        yaml_file = "config/stagn_cfg.yaml"
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


def base_load_data(args: dict):
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args["test_size"]

    if args["method"] == "stan":
        features, labels = span_data_3d(feat_df)
    else:
        features, labels = span_data_2d(feat_df)

    trf, tef, trl, tel = train_test_split(
        features,
        labels,
        train_size=train_size,
        stratify=labels,
        shuffle=True,
    )

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

        elif args["method"] == "rgtan":
            from methods.rgtan.rgtan_main import rgtan_main, loda_rgtan_data

            feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
                dataset=args["dataset"],
                test_size=args["test_size"],
                ieee_mode=args["ieee_mode"],
            )

            rgtan_main(
                feat_data,
                g,
                train_idx,
                test_idx,
                labels,
                args,
                cat_features,
                neigh_features,
                experiment=experiment,
            )

        elif args["method"] == "hogrl":
            from methods.hogrl.hogrl_main import hogrl_main
            hogrl_main(args)

        if experiment:
            experiment.log_other("status", "success")

    except Exception as e:
        if experiment:
            experiment.log_other("status", "failed")
            experiment.log_other("exception", repr(e))
        raise

    finally:
        if experiment:
            print("Ending Comet experiment...")
            experiment.end()


if __name__ == "__main__":
    main(parse_args())
