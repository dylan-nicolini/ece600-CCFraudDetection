import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d

logger = logging.getLogger(__name__)

# --- Optional Comet support ---
try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


def init_comet(args: dict):
    """
    Create ONE Comet experiment per run.

    Requires these env vars:
      - COMET_API_KEY
      - COMET_WORKSPACE
      - COMET_PROJECT_NAME (optional; default used if missing)

    Returns:
      - Experiment instance, or None if disabled/missing deps.
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

    # Helpful metadata
    try:
        exp.set_name(f"{args.get('method', 'run')}-{args.get('dataset', 'na')}-seed{args.get('seed', 'na')}")
        exp.add_tag(str(args.get("method", "unknown")))
        if "dataset" in args:
            exp.add_tag(str(args["dataset"]))
        exp.log_parameters(args)
    except Exception:
        pass

    return exp


def log_comet_status(experiment):
    """
    Print Comet status diagnostics immediately at startup,
    so you can confirm the experiment should appear in Comet UI.
    """
    print("----- COMET STATUS -----")

    api_key = os.getenv("COMET_API_KEY")
    workspace = os.getenv("COMET_WORKSPACE")
    project = os.getenv("COMET_PROJECT_NAME")

    print("COMET_API_KEY set:", "YES" if api_key else "NO")
    if api_key:
        print("COMET_API_KEY preview:", api_key[:6] + "..." + api_key[-4:])

    print("COMET_WORKSPACE:", workspace)
    print("COMET_PROJECT_NAME:", project)

    if experiment is None:
        print("Experiment object: None (Comet disabled / not initialized)")
    else:
        print("Experiment object created:", True)
        try:
            print("Experiment disabled:", experiment.disabled)
        except Exception:
            print("Experiment disabled: (unknown)")

    print("------------------------")


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve"
    )
    parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())["method"]

    if method in ["mcnn"]:
        yaml_file = "config/mcnn_cfg.yaml"
    elif method in ["stan"]:
        yaml_file = "config/stan_cfg.yaml"
    elif method in ["stan_2d"]:
        yaml_file = "config/stan_2d_cfg.yaml"
    elif method in ["stagn"]:
        yaml_file = "config/stagn_cfg.yaml"
    elif method in ["gtan"]:
        yaml_file = "config/gtan_cfg.yaml"
    elif method in ["rgtan"]:
        yaml_file = "config/rgtan_cfg.yaml"
    elif method in ["hogrl"]:
        yaml_file = "config/hogrl_cfg.yaml"
    else:
        raise NotImplementedError("Unsupported method.")

    with open(yaml_file, "r") as file:
        args = yaml.safe_load(file)

    args["method"] = method
    return args


def base_load_data(args: dict):
    """
    Load S-FFSD dataset for base models (non-graph baselines).
    Writes npy outputs for the base methods.
    """
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args["test_size"]

    # for ICONIP16 & AAAI20
    if args["method"] == "stan":
        if os.path.exists("data/tel_3d.npy"):
            return
        features, labels = span_data_3d(feat_df)
    else:
        if os.path.exists("data/tel_2d.npy"):
            return
        features, labels = span_data_2d(feat_df)

    trf, tef, trl, tel = train_test_split(
        features,
        labels,
        train_size=train_size,
        stratify=labels,
        shuffle=True
    )

    trf_file, tef_file, trl_file, tel_file = (
        args["trainfeature"],
        args["testfeature"],
        args["trainlabel"],
        args["testlabel"]
    )

    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)


def main(args):
    experiment = init_comet(args)

    # 🔍 Startup diagnostics: you should see these BEFORE training
    log_comet_status(experiment)

    try:
        if args["method"] == "mcnn":
            from methods.mcnn.mcnn_main import mcnn_main
            base_load_data(args)
            mcnn_main(
                args["trainfeature"],
                args["trainlabel"],
                args["testfeature"],
                args["testlabel"],
                epochs=args["epochs"],
                batch_size=args["batch_size"],
                lr=args["lr"],
                device=args["device"]
            )

        elif args["method"] == "stan_2d":
            from methods.stan.stan_2d_main import stan_main
            base_load_data(args)
            stan_main(
                args["trainfeature"],
                args["trainlabel"],
                args["testfeature"],
                args["testlabel"],
                mode="2d",
                epochs=args["epochs"],
                batch_size=args["batch_size"],
                attention_hidden_dim=args["attention_hidden_dim"],
                lr=args["lr"],
                device=args["device"]
            )

        elif args["method"] == "stan":
            from methods.stan.stan_main import stan_main
            base_load_data(args)
            stan_main(
                args["trainfeature"],
                args["trainlabel"],
                args["testfeature"],
                args["testlabel"],
                mode="3d",
                epochs=args["epochs"],
                batch_size=args["batch_size"],
                attention_hidden_dim=args["attention_hidden_dim"],
                lr=args["lr"],
                device=args["device"]
            )

        elif args["method"] == "stagn":
            from methods.stagn.stagn_main import stagn_main, load_stagn_data
            features, labels, g = load_stagn_data(args)
            stagn_main(
                features,
                labels,
                args["test_size"],
                g,
                mode="2d",
                epochs=args["epochs"],
                attention_hidden_dim=args["attention_hidden_dim"],
                lr=args["lr"],
                device=args["device"]
            )

        elif args["method"] == "gtan":
            from methods.gtan.gtan_main import gtan_main, load_gtan_data
            feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
                args["dataset"], args["test_size"]
            )

            # ✅ Pass Comet experiment if gtan_main supports it; else fall back.
            try:
                gtan_main(feat_data, g, train_idx, test_idx, labels, args, cat_features, experiment=experiment)
            except TypeError:
                gtan_main(feat_data, g, train_idx, test_idx, labels, args, cat_features)

        elif args["method"] == "rgtan":
            from methods.rgtan.rgtan_main import rgtan_main, loda_rgtan_data

            feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
                args["dataset"], args["test_size"]
            )

            # Enforce GTAN-standard Comet logging for RGTAN (no silent fallback)
            rgtan_main(
                feat_data, g, train_idx, test_idx, labels, args,
                cat_features, neigh_features,
                nei_att_head=args["nei_att_heads"][args["dataset"]],
                experiment=experiment
            )


        elif args["method"] == "hogrl":
            from methods.hogrl.hogrl_main import hogrl_main
            hogrl_main(args)

        else:
            raise NotImplementedError("Unsupported method.")

        if experiment:
            try:
                experiment.log_other("status", "success")
            except Exception:
                pass

    except Exception as e:
        if experiment:
            try:
                experiment.log_other("status", "failed")
                experiment.log_other("exception", repr(e))
            except Exception:
                pass
        raise

    finally:
        if experiment:
            print("Ending Comet experiment...")
            experiment.end()
            print("Comet experiment ended.")


if __name__ == "__main__":
    main(parse_args())
