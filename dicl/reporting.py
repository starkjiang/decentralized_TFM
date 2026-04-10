"""
dicl/reporting.py
==================
Console summary tables and JSON serialisation.

Tables
------
  print_clf_table      — classification results (accuracy, F1, gap vs oracle)
  print_reg_table      — regression results (RMSE, R²)
  print_ablation_table — generic ablation results

JSON
----
  save_results_json    — writes clf_res, reg_res, ablations to a single file

JSON schema
-----------
{
  "classification": [ <run_dict>, ... ],
  "regression":     [ <run_dict>, ... ],
  "ablations": {
    "topology":  [...],
    "consensus": [...],
    "tau":       [...],
    "K":         [...],
    "alpha":     [...],
  }
}
"""

import json
import os

import numpy as np


# =============================================================================
# Console tables
# =============================================================================

def print_clf_table(results: list) -> None:
    H = (
        "  "
        + f"{'Dataset':>24}|{'BB':>8}|{'K':>3}|{'Part':>8}"
        + f"|{'SA':>6}|{'D0':>6}|{'DT':>6}|{'Centr':>6}|{'F1@T':>6}|{'Gap':>6}"
    )
    sep = "═" * len(H)
    print(f"\n{sep}")
    print("  CLASSIFICATION  (SA=single-agent  Centr=centralised  DT=D-ICL@T)")
    print(f"{sep}\n{H}\n  {'─' * (len(H) - 2)}")
    for r in results:
        sa    = r["single_agent"]["accuracy"]
        d0    = r["baseline"]["consensus"]["accuracy"]
        dT    = r["final"]["consensus"]["accuracy"]
        centr = r["oracle"]["accuracy"]
        f1    = r["final"]["consensus"]["f1_macro"]
        print(
            f"  {r['dataset']:>24}|{r['backbone']:>8}|{r['K']:>3}"
            f"|{r['partition']:>8}|{sa:>6.3f}|{d0:>6.3f}|{dT:>6.3f}"
            f"|{centr:>6.3f}|{f1:>6.3f}|{dT - centr:>+6.3f}"
        )
    print(sep)


def print_reg_table(results: list) -> None:
    H = (
        "  "
        + f"{'Dataset':>26}|{'BB':>8}|{'Part':>8}"
        + f"|{'SA RMSE':>9}|{'D-ICL@0':>9}|{'D-ICL@T':>9}|{'Centr':>9}|{'R2@T':>6}"
    )
    sep = "═" * len(H)
    print(f"\n{sep}")
    print("  REGRESSION  (RMSE lower is better; K=4)")
    print(f"{sep}\n{H}\n  {'─' * (len(H) - 2)}")
    for r in results:
        sa    = r["single_agent"]["rmse"]
        d0    = r["baseline"]["consensus"]["rmse"]
        dT    = r["final"]["consensus"]["rmse"]
        centr = r["oracle"]["rmse"]
        r2    = r["final"]["consensus"]["r2"]
        print(
            f"  {r['dataset']:>26}|{r['backbone']:>8}|{r['partition']:>8}"
            f"|{sa:>9.4f}|{d0:>9.4f}|{dT:>9.4f}|{centr:>9.4f}|{r2:>6.3f}"
        )
    print(sep)


def print_ablation_table(results: list, name: str, key: str) -> None:
    print(f"\n  ABLATION: {name}")
    print(
        f"  {'BB':>8}|{key:>12}"
        f"|{'SA':>6}|{'D-ICL@0':>8}|{'D-ICL@T':>8}|{'Centr':>6}|{'Gap':>6}"
    )
    print("  " + "─" * 56)
    for r in results:
        sa    = r["single_agent"]["accuracy"]
        d0    = r["baseline"]["consensus"]["accuracy"]
        dT    = r["final"]["consensus"]["accuracy"]
        centr = r["oracle"]["accuracy"]
        val   = r.get(key, "?")
        print(
            f"  {r['backbone']:>8}|{str(val):>12}"
            f"|{sa:>6.3f}|{d0:>8.3f}|{dT:>8.3f}|{centr:>6.3f}|{dT - centr:>+6.3f}"
        )


# =============================================================================
# JSON serialisation
# =============================================================================

def _serialise(obj):
    """Recursively convert numpy scalars / arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def save_results_json(
    clf_res: list,
    reg_res: list,
    abl:     dict,
    path:    str = "dicl_results.json",
) -> None:
    """Save all experimental results to a single structured JSON file."""
    payload = {
        "classification": _serialise(clf_res),
        "regression":     _serialise(reg_res),
        "ablations":      _serialise(abl),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Results saved → {path}  ({size_kb:.1f} KB)")
