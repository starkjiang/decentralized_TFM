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

def _fmt(val, std=None):
    """Format a metric value, appending ±std when available and finite."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  n/a "
    s = f"{val:.3f}"
    if std is not None and not np.isnan(std) and std > 0:
        s += f"±{std:.3f}"
    return s


def print_clf_table(results):
    """
    Print classification results.  If results are seeded aggregates the
    accuracy columns show mean±std; otherwise plain mean.
    """
    seeded = any("n_seeds" in r for r in results)
    hdr = ("  " + f"{'Dataset':>24}|{'BB':>8}|{'K':>3}|{'Part':>8}|"
           + f"{'SA':>12}|{'FedAvg':>12}|{'D-ICL@0':>12}|"
           + f"{'D-ICL@T':>12}|{'Centr':>12}|{'Gap':>7}")
    print("\n" + "═"*len(hdr))
    tag = f"  CLASSIFICATION  ({'mean±std, ' if seeded else ''}SA=single-agent  Centr=centralized)"
    print(tag); print("═"*len(hdr)); print(hdr); print("  " + "─"*(len(hdr)-2))
    for r in results:
        def _g(d, k):   return d.get(k, float("nan"))
        def _gs(d, k):  return d.get(k+"_std", float("nan")) if seeded else None
        sa   = _fmt(_g(r["single_agent"],"accuracy"),  _gs(r["single_agent"],"accuracy"))
        fa   = _fmt(_g(r.get("fedavg",{}),"accuracy"), _gs(r.get("fedavg",{}),"accuracy"))
        d0   = _fmt(_g(r["baseline"]["consensus"],"accuracy"),
                    _gs(r["baseline"]["consensus"],"accuracy"))
        dT   = _fmt(_g(r["final"]["consensus"],"accuracy"),
                    _gs(r["final"]["consensus"],"accuracy"))
        cen  = _fmt(_g(r["oracle"],"accuracy"), _gs(r["oracle"],"accuracy"))
        gap  = _g(r["final"]["consensus"],"accuracy") - _g(r["oracle"],"accuracy")
        print(f"  {r['dataset']:>24}|{r['backbone']:>8}|{r['K']:>3}|{r['partition']:>8}|"
              f"{sa:>12}|{fa:>12}|{d0:>12}|{dT:>12}|{cen:>12}|{gap:>+7.3f}")
    print("═"*len(hdr))


def print_reg_table(results):
    """
    Print regression results with mean±std when seeded.
    """
    seeded = any("n_seeds" in r for r in results)
    hdr = ("  " + f"{'Dataset':>26}|{'BB':>8}|{'Part':>8}|"
           + f"{'SA RMSE':>12}|{'FedAvg':>12}|{'D-ICL@0':>12}|"
           + f"{'D-ICL@T':>12}|{'Centr':>12}|{'R2@T':>8}")
    print("\n" + "═"*len(hdr))
    tag = f"  REGRESSION  ({'mean±std, ' if seeded else ''}RMSE lower is better; K=4)"
    print(tag); print("═"*len(hdr)); print(hdr); print("  " + "─"*(len(hdr)-2))
    for r in results:
        def _g(d, k):   return d.get(k, float("nan"))
        def _gs(d, k):  return d.get(k+"_std", float("nan")) if seeded else None
        sa   = _fmt(_g(r["single_agent"],"rmse"), _gs(r["single_agent"],"rmse"))
        fa   = _fmt(_g(r.get("fedavg",{}),"rmse"),_gs(r.get("fedavg",{}),"rmse"))
        d0   = _fmt(_g(r["baseline"]["consensus"],"rmse"),
                    _gs(r["baseline"]["consensus"],"rmse"))
        dT   = _fmt(_g(r["final"]["consensus"],"rmse"),
                    _gs(r["final"]["consensus"],"rmse"))
        cen  = _fmt(_g(r["oracle"],"rmse"), _gs(r["oracle"],"rmse"))
        r2   = _fmt(_g(r["final"]["consensus"],"r2"),
                    _gs(r["final"]["consensus"],"r2"))
        print(f"  {r['dataset']:>26}|{r['backbone']:>8}|{r['partition']:>8}|"
              f"{sa:>12}|{fa:>12}|{d0:>12}|{dT:>12}|{cen:>12}|{r2:>8}")
    print("═"*len(hdr))

def print_ablation_table(results,name,key):
    print(f"\n  ABLATION: {name}")
    print(f"  {'BB':>8}|{key:>12}|{'SA':>6}|{'D-ICL@0':>8}|{'D-ICL@T':>8}|{'Centr':>6}|{'Gap':>6}")
    print("  "+"─"*56)
    for r in results:
        sa=r["single_agent"]["accuracy"]; d0=r["baseline"]["consensus"]["accuracy"]
        dT=r["final"]["consensus"]["accuracy"]; centr=r["oracle"]["accuracy"]
        val=r.get(key,"?")
        print(f"  {r['backbone']:>8}|{str(val):>12}|{sa:>6.3f}|{d0:>8.3f}|{dT:>8.3f}|{centr:>6.3f}|{dT-centr:>+6.3f}")


# =============================================================================
# JSON serialisation
# =============================================================================

def _serialise(obj):
    """Recursively convert numpy scalars/arrays to plain Python for JSON."""
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
    # Drop non-serialisable objects (e.g. sklearn scalers)
    try:
        import json; json.dumps(obj); return obj
    except (TypeError, ValueError):
        return str(obj)


def save_results_json(clf_res, reg_res, abl, path="dicl_results.json"):
    """
    Save all experimental results to a single structured JSON file.

    Schema
    ------
    {
      "classification": [ <run_dict>, ... ],
      "regression":     [ <run_dict>, ... ],
      "ablations": {
        "topology":  [ <run_dict>, ... ],
        "consensus": [ ... ],
        "tau":       [ ... ],
        "K":         [ ... ],
        "alpha":     [ ... ],
      }
    }

    Each run_dict contains dataset, backbone, partition, K, topology,
    consensus, tau, alpha, single_agent metrics, oracle metrics, and a
    list of per-round metrics.
    """
    payload = {
        "classification": _serialise(clf_res),
        "regression":     _serialise(reg_res),
        "ablations":      _serialise(abl),
    }
    import json
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Results saved → {path}  ({size_kb:.1f} KB)")
