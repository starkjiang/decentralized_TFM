"""
main.py
=======
Entry point for D-ICL experiments.

Usage
-----
    python main.py                            # full run (all datasets, all ablations)
    python main.py --rounds 2                 # quick smoke-test
    python main.py --datasets breast_cancer   # single clf dataset
    python main.py --no-ablations             # skip ablation studies
    python main.py --output ./my_results      # custom output path
"""

import argparse
import os
import warnings
import random

warnings.filterwarnings("ignore")

import numpy as np
import torch

from dicl import (
    Config,
    load_clf, load_reg,
    run_main,
    run_ablation_topology, run_ablation_consensus,
    run_ablation_tau, run_ablation_K, run_ablation_alpha,
    print_clf_table, print_reg_table, print_ablation_table,
    save_results_json,
    build_figures,
)

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D-ICL experiments")
    p.add_argument(
        "--clf-datasets", nargs="+",
        default=None,
        help="Classification datasets to run (default: all 5)",
    )
    p.add_argument(
        "--reg-datasets", nargs="+",
        default=None,
        help="Regression datasets to run (default: all 5)",
    )
    p.add_argument(
        "--backbones", nargs="+",
        choices=["tabicl", "tabpfn"],
        default=None,
        help="Backbones to use (default: tabicl tabpfn)",
    )
    p.add_argument(
        "--rounds", type=int, default=None,
        help="Override number of D-ICL rounds T (e.g. 2 for a smoke-test)",
    )
    p.add_argument(
        "--no-ablations", action="store_true",
        help="Skip all ablation studies",
    )
    p.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation",
    )
    p.add_argument(
        "--output", type=str, default="dicl_results.json",
        help="Path for the results JSON file",
    )
    p.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Seed ─────────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = Config()
    if args.clf_datasets:
        cfg.clf_datasets = args.clf_datasets
    if args.reg_datasets:
        cfg.reg_datasets = args.reg_datasets
    if args.backbones:
        cfg.backbones = args.backbones
    if args.rounds is not None:
        cfg.T = args.rounds

    if args.no_ablations:
        cfg.run_ablation_topology  = False
        cfg.run_ablation_consensus = False
        cfg.run_ablation_tau       = False
        cfg.run_ablation_K         = False
        cfg.run_ablation_alpha     = False

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\n" + "═" * 74 + "\n  LOADING DATASETS\n" + "═" * 74)
    all_clf = set(cfg.clf_datasets) | {cfg.ablation_dataset}
    clf_cache = {ds: load_clf(ds, cfg) for ds in all_clf}
    reg_cache = {ds: load_reg(ds, cfg) for ds in cfg.reg_datasets}

    # ── Main experiments ──────────────────────────────────────────────────────
    print("\n" + "═" * 74 + "\n  MAIN EXPERIMENTS\n" + "═" * 74)
    clf_res, reg_res = run_main(cfg, clf_cache, reg_cache)
    print_clf_table(clf_res)
    print_reg_table(reg_res)

    # ── Ablations ─────────────────────────────────────────────────────────────
    abl = {}
    ablation_schedule = [
        (cfg.run_ablation_topology,  "TOPOLOGY",  run_ablation_topology),
        (cfg.run_ablation_consensus, "CONSENSUS", run_ablation_consensus),
        (cfg.run_ablation_tau,       "TAU",       run_ablation_tau),
        (cfg.run_ablation_K,         "K",         run_ablation_K),
        (cfg.run_ablation_alpha,     "ALPHA",     run_ablation_alpha),
    ]
    for flag, name, fn in ablation_schedule:
        if flag:
            print("\n" + "═" * 74 + f"\n  ABLATION: {name}\n" + "═" * 74)
            key        = name.lower()
            abl[key]   = fn(cfg, clf_cache)
            abl_key    = key if key != "topology" else "topology"
            print_ablation_table(abl[key], name, abl_key)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 74 + "\n  SAVING RESULTS (JSON)\n" + "═" * 74)
    save_results_json(clf_res, reg_res, abl, path=args.output)

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.no_figures:
        print("\n" + "═" * 74 + "\n  BUILDING FIGURES\n" + "═" * 74)
        build_figures(clf_res, reg_res, cfg)

    print("\n" + "═" * 74 + "\n  DONE\n" + "═" * 74)


if __name__ == "__main__":
    main()
