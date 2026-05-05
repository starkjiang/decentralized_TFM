"""
dicl/runner.py
==============
Evaluation metrics, single-agent / centralised baselines,
the D-ICL training loops for classification and regression,
the main experiment suite, and all ablation runners.

D-ICL protocol (per round t)
-----------------------------
  Phase 1 — Local ICL Inference:
      Each agent queries its backbone f_θ(x | C_k^t) on the test set and
      the query pool.
  Phase 3 — Consensus:
      Per-agent neighbourhood predictions are aggregated via the chosen
      consensus function (arithmetic / weighted / geometric).
  Phase 4 — Context Enrichment:
      High-confidence pseudo-labels (classification: conf ≥ τ;
      regression: low inter-agent std) are appended to each agent's
      context reservoir.
"""

import itertools
import time
from copy import deepcopy
from typing import Any, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
)

from .config import Config
from .agents import make_clf_agent, make_reg_agent
from .data   import partition_iid, partition_noniid
from .topology import TOPOLOGIES, CONSENSUS_FNS, aggregate_all

# ── Types ─────────────────────────────────────────────────────────────────────
RunResult = dict[str, Any]


# =============================================================================
# Metrics
# =============================================================================

def eval_clf(y_true, proba, n_cls) -> dict:
    """Return accuracy, macro-F1, ROC-AUC, and log-loss."""
    yp  = proba.argmax(1)
    acc = float(accuracy_score(y_true, yp))
    f1  = float(f1_score(y_true, yp, average="macro", zero_division=0))
    ll  = float(log_loss(y_true, proba + 1e-12, labels=list(range(n_cls))))
    try:
        auc = (
            float(roc_auc_score(y_true, proba[:, 1]))
            if n_cls == 2
            else float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        )
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "f1_macro": f1, "roc_auc": auc, "log_loss": ll}


def eval_reg(y_true, y_pred) -> dict:
    """Return RMSE, MAE, and R²."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# =============================================================================
# Baselines
# =============================================================================

def single_agent_clf(Xtr, ytr, Xte, yte, backbone, cfg, meta) -> dict:
    """Single-agent baseline: one agent trained on all training data."""
    n_cls = meta["n_classes"]
    a     = make_clf_agent(backbone, 0, Xtr, ytr, n_cls, cfg)
    return eval_clf(yte, a.predict_proba(Xte), n_cls)


def centralised_clf(Xtr, ytr, Xte, yte, backbone, cfg, meta) -> dict:
    """Centralised oracle: one agent with m_max training examples as context."""
    n_cls = meta["n_classes"]
    n_ctx = min(cfg.m_max, len(Xtr))
    idx   = np.random.choice(len(Xtr), n_ctx, replace=False)
    a     = make_clf_agent(backbone, 0, Xtr[idx], ytr[idx], n_cls, cfg)
    a.C_x = Xtr[idx]; a.C_y = ytr[idx]; a._refresh_context()
    return eval_clf(yte, a.predict_proba(Xte), n_cls)


def single_agent_reg(Xtr, ytr, Xte, yte, backbone, cfg, meta) -> dict:
    """Single-agent regression baseline."""
    a = make_reg_agent(backbone, 0, Xtr, ytr, cfg)
    return eval_reg(yte, a.predict(Xte))


def centralised_reg(Xtr, ytr, Xte, yte, backbone, cfg, meta) -> dict:
    """Centralised oracle regression baseline."""
    n_ctx = min(cfg.m_max, len(Xtr))
    idx   = np.random.choice(len(Xtr), n_ctx, replace=False)
    a     = make_reg_agent(backbone, 0, Xtr[idx], ytr[idx], cfg)
    a.C_x = Xtr[idx]; a.C_y = ytr[idx]; a._refresh_context()
    return eval_reg(yte, a.predict(Xte))


def fedavg_clf(Xtr, ytr, Xte, yte, K, partition, cfg, meta, backbone, T=None):
    """
    FedAvg classification baseline using the same TFM backbone as D-ICL.

    Each agent is initialised identically to D-ICL round 0: stratified sample
    of size m_0 from its private shard.  The server averages the K probability
    vectors on the test set.  No context enrichment is performed across rounds,
    so the flat ensemble accuracy is constant and is reported for every round
    to keep the result shape consistent with D-ICL.

    Parameters
    ----------
    backbone : str   "tabicl" or "tabpfn" — same backbone as the D-ICL run
                     being compared.
    """
    n_cls  = meta["n_classes"]
    T_     = T if T is not None else cfg.T
    alpha_ = cfg.alpha_dirichlet

    parts = (partition_iid(Xtr, ytr, K) if partition == "iid"
             else partition_noniid(Xtr, ytr, K, alpha_))

    # Build K agents with identical initialisation to D-ICL round 0
    agents = [make_clf_agent(backbone, k, Xk, yk, n_cls, cfg)
              for k, (Xk, yk) in enumerate(parts)]

    # One-shot inference: average probability vectors across all K agents
    try:
        proba_list = [a.predict_proba(Xte) for a in agents]
        avg_proba  = np.mean(np.stack(proba_list, axis=0), axis=0)  # (N, C)
        metrics    = eval_clf(yte, avg_proba, n_cls)
    except Exception:
        metrics = {"accuracy": float("nan"), "f1_macro": float("nan"),
                   "roc_auc": float("nan"), "log_loss": float("nan")}

    # Repeat the same value for every round (no improvement — no enrichment)
    rounds_acc = [metrics["accuracy"]] * (T_ + 1)

    return {
        "accuracy":   metrics["accuracy"],
        "f1_macro":   metrics["f1_macro"],
        "roc_auc":    metrics["roc_auc"],
        "log_loss":   metrics["log_loss"],
        "rounds_acc": rounds_acc,
    }


def fedavg_reg(Xtr, ytr, Xte, yte, K, partition, cfg, meta, backbone, T=None):
    """
    FedAvg regression baseline using the same TFM backbone as D-ICL.

    Identical logic to fedavg_clf but for regression: each agent predicts
    scalar values; the server averages them.  No context enrichment.
    """
    T_     = T if T is not None else cfg.T
    alpha_ = cfg.alpha_dirichlet

    parts = (partition_iid(Xtr, ytr, K) if partition == "iid"
             else partition_noniid(Xtr, ytr, K, alpha_))

    agents = [make_reg_agent(backbone, k, Xk, yk, cfg)
              for k, (Xk, yk) in enumerate(parts)]

    try:
        pred_list = [a.predict(Xte) for a in agents]
        avg_pred  = np.mean(np.stack(pred_list, axis=0), axis=0)  # (N,)
        metrics   = eval_reg(yte, avg_pred)
    except Exception:
        metrics = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    rounds_rmse = [metrics["rmse"]] * (T_ + 1)

    return {
        "rmse":        metrics["rmse"],
        "mae":         metrics["mae"],
        "r2":          metrics["r2"],
        "rounds_rmse": rounds_rmse,
    }


# =============================================================================
# D-ICL Classification
# =============================================================================

def run_dicl_clf(
    Xtr, ytr, Xte, yte,
    K:         int,
    partition: str,
    backbone:  str,
    cfg:       Config,
    meta:      dict,
    topology:  str   = "fully_connected",
    consensus: str   = "arithmetic",
    tau:       float | None = None,
    alpha:     float | None = None,
    verbose:   bool  = True,
) -> RunResult:
    """
    Run D-ICL classification for T rounds and return a structured result dict.

    Result schema
    -------------
    dataset, short, K, partition, backbone, topology, consensus, tau, alpha,
    single_agent, oracle, rounds (list), baseline (round 0), final (round T).
    """
    tau   = tau   or cfg.tau
    alpha = alpha or cfg.alpha_dirichlet

    cfg_l     = deepcopy(cfg)
    cfg_l.tau = tau
    n_cls     = meta["n_classes"]

    if verbose:
        print(
            f"\n  {'─'*68}\n  D-ICL-CLF | {meta['name'][:22]} | K={K} | "
            f"{partition} | {backbone} | topo={topology} | τ={tau:.2f}\n  {'─'*68}"
        )

    parts  = (
        partition_iid(Xtr, ytr, K)
        if partition == "iid"
        else partition_noniid(Xtr, ytr, K, alpha, m_0=cfg.m_0)
    )
    A      = TOPOLOGIES[topology](K)
    agents = [make_clf_agent(backbone, k, Xk, yk, n_cls, cfg_l)
              for k, (Xk, yk) in enumerate(parts)]

    sa_m     = single_agent_clf(Xtr, ytr, Xte, yte, backbone, cfg, meta)
    centr_m  = centralised_clf(Xtr, ytr, Xte, yte, backbone, cfg, meta)

    n_pool = min(max(32, int(len(Xtr) * cfg.query_pool_frac)), len(Xtr))
    Xpool  = Xtr[np.random.choice(len(Xtr), n_pool, replace=False)]

    res = {
        "dataset": meta["name"], "short": meta["short"],
        "K": K, "partition": partition, "backbone": backbone,
        "topology": topology, "consensus": consensus,
        "tau": tau, "alpha": alpha,
        "rounds": [],
        "single_agent": sa_m, "oracle": centr_m,
    }

    for t in range(cfg_l.T + 1):
        t0        = time.time()
        prob_test = [a.predict_proba(Xte) for a in agents]
        prob_pool = ([a.predict_proba(Xpool) for a in agents] if t < cfg_l.T else [])

        bar_test  = aggregate_all(prob_test, A, consensus)
        cons_m    = eval_clf(yte, bar_test[0], n_cls)
        ctx_sizes = [a.context_size for a in agents]

        rec = {
            "round":        t,
            "elapsed":      round(time.time() - t0, 2),
            "consensus":    cons_m,
            "context_sizes": ctx_sizes,
            "oracle_gap":   round(centr_m["accuracy"] - cons_m["accuracy"], 4),
            "sa_gain":      round(cons_m["accuracy"] - sa_m["accuracy"], 4),
        }
        res["rounds"].append(rec)

        if verbose:
            tag = "  Base" if t == 0 else f"  Rd {t:2d}"
            print(
                f"{tag}  D-ICL={cons_m['accuracy']:.4f}  "
                f"SA={sa_m['accuracy']:.4f}  Centr={centr_m['accuracy']:.4f}  "
                f"Gap={rec['oracle_gap']:+.4f}"
            )

        if t == cfg_l.T:
            break

        bar_pool = aggregate_all(prob_pool, A, consensus)
        n_added  = sum(
            agents[k].update_context(Xpool, bar_pool[k]) for k in range(K)
        )
        if verbose:
            print(
                f"         +{n_added} pseudo-labels  "
                f"ctx: {np.mean(ctx_sizes):.0f}→"
                f"{np.mean([a.context_size for a in agents]):.0f}"
            )

    res["baseline"] = res["rounds"][0]
    res["final"]    = res["rounds"][-1]
    return res


# =============================================================================
# D-ICL Regression
# =============================================================================

def run_dicl_reg(
    Xtr, ytr, Xte, yte,
    K:         int,
    partition: str,
    backbone:  str,
    cfg:       Config,
    meta:      dict,
    topology:  str   = "fully_connected",
    alpha:     float | None = None,
    verbose:   bool  = True,
) -> RunResult:
    """
    Run D-ICL regression for T rounds and return a structured result dict.

    Consensus for regression is always arithmetic mean over the neighbourhood.
    Context enrichment selects pool points with low inter-agent prediction std.
    """
    alpha = alpha or cfg.alpha_dirichlet
    cfg_l = deepcopy(cfg)

    if verbose:
        print(
            f"\n  {'─'*68}\n  D-ICL-REG | {meta['name'][:22]} | K={K} | "
            f"{partition} | {backbone} | topo={topology}\n  {'─'*68}"
        )

    parts  = (
        partition_iid(Xtr, ytr, K)
        if partition == "iid"
        else partition_noniid(Xtr, ytr, K, alpha, m_0=cfg.m_0)
    )
    A      = TOPOLOGIES[topology](K)
    agents = [make_reg_agent(backbone, k, Xk, yk, cfg_l)
              for k, (Xk, yk) in enumerate(parts)]

    sa_m    = single_agent_reg(Xtr, ytr, Xte, yte, backbone, cfg, meta)
    centr_m = centralised_reg(Xtr, ytr, Xte, yte, backbone, cfg, meta)

    n_pool = min(max(32, int(len(Xtr) * cfg.query_pool_frac)), len(Xtr))
    Xpool  = Xtr[np.random.choice(len(Xtr), n_pool, replace=False)]

    res = {
        "dataset": meta["name"], "short": meta["short"],
        "K": K, "partition": partition, "backbone": backbone,
        "topology": topology, "alpha": alpha,
        "rounds": [],
        "single_agent": sa_m, "oracle": centr_m,
    }

    for t in range(cfg_l.T + 1):
        t0        = time.time()
        pred_test = [a.predict(Xte) for a in agents]
        pred_pool = ([a.predict(Xpool) for a in agents] if t < cfg_l.T else [])

        bar_test = []
        for k in range(K):
            nb = np.where(A[k] > 0)[0]
            bar_test.append(np.stack([pred_test[j] for j in nb]).mean(0))

        cons_m    = eval_reg(yte, bar_test[0])
        ctx_sizes = [a.context_size for a in agents]

        rec = {
            "round":          t,
            "elapsed":        round(time.time() - t0, 2),
            "consensus":      cons_m,
            "context_sizes":  ctx_sizes,
            "oracle_gap_rmse": round(cons_m["rmse"] - centr_m["rmse"], 4),
            "sa_gain_rmse":    round(sa_m["rmse"] - cons_m["rmse"], 4),
        }
        res["rounds"].append(rec)

        if verbose:
            tag = "  Base" if t == 0 else f"  Rd {t:2d}"
            print(
                f"{tag}  D-ICL RMSE={cons_m['rmse']:.4f}  R2={cons_m['r2']:.4f}  "
                f"SA={sa_m['rmse']:.4f}  Centr={centr_m['rmse']:.4f}"
            )

        if t == cfg_l.T:
            break

        bar_pool = []
        for k in range(K):
            nb = np.where(A[k] > 0)[0]
            bar_pool.append(np.stack([pred_pool[j] for j in nb]).mean(0))

        n_added = sum(
            agents[k].update_context_reg(Xpool, bar_pool[k], pred_pool)
            for k in range(K)
        )
        if verbose:
            print(
                f"         +{n_added} pseudo-labels  "
                f"ctx: {np.mean(ctx_sizes):.0f}→"
                f"{np.mean([a.context_size for a in agents]):.0f}"
            )

    res["baseline"] = res["rounds"][0]
    res["final"]    = res["rounds"][-1]
    return res


# =============================================================================
# Main experiment suite
# =============================================================================

def run_main(
    cfg:       Config,
    clf_cache: dict,
    reg_cache: dict,
) -> tuple[list[RunResult], list[RunResult]]:
    """
    Run all classification × {IID, non-IID} × K × backbone conditions,
    then all regression × {IID, non-IID} × backbone conditions.
    """
    clf_res: list[RunResult] = []
    conds   = list(itertools.product(
        cfg.clf_datasets, ["iid", "non_iid"], cfg.K_values, cfg.backbones
    ))
    for idx, (ds, part, K, bb) in enumerate(conds, 1):
        Xtr, Xte, ytr, yte, meta = clf_cache[ds]
        print(f"\n[CLF {idx:3d}/{len(conds)}]", end="")
        clf_res.append(
            run_dicl_clf(Xtr, ytr, Xte, yte, K=K, partition=part,
                         backbone=bb, cfg=cfg, meta=meta, verbose=True)
        )

    reg_res: list[RunResult] = []
    K_reg   = 4
    conds_r = list(itertools.product(cfg.reg_datasets, ["iid", "non_iid"], cfg.backbones))
    for idx, (ds, part, bb) in enumerate(conds_r, 1):
        Xtr, Xte, ytr, yte, meta = reg_cache[ds]
        print(f"\n[REG {idx:3d}/{len(conds_r)}]", end="")
        reg_res.append(
            run_dicl_reg(Xtr, ytr, Xte, yte, K=K_reg, partition=part,
                         backbone=bb, cfg=cfg, meta=meta, verbose=True)
        )

    return clf_res, reg_res


# ══════════════════════════════════════════════════════════════════════════════
# Muti-seed evaluation and aggregation
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Re-seed all random sources for a new trial."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _agg_metrics(runs_key: List[dict]) -> dict:
    """
    Given a list of metric dicts (one per seed), return a dict with:
        key        : mean across seeds
        key_std    : std  across seeds
    for every numeric value in the dict.
    """
    if not runs_key:
        return {}
    keys = runs_key[0].keys()
    out  = {}
    for k in keys:
        vals = [r[k] for r in runs_key if isinstance(r.get(k), (int, float))
                and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if vals:
            out[k]           = float(np.mean(vals))
            out[k + "_std"]  = float(np.std(vals, ddof=0))
        else:
            out[k]          = float("nan")
            out[k + "_std"] = float("nan")
    return out


def _agg_rounds(rounds_per_seed: List[List[dict]]) -> List[dict]:
    """
    Aggregate per-round records across seeds.
    rounds_per_seed[seed_idx][round_idx] → dict with "consensus", etc.
    Returns a list of round dicts where every numeric leaf has a _std sibling.
    """
    n_rounds = len(rounds_per_seed[0])
    agg = []
    for t in range(n_rounds):
        # Collect consensus metric dicts for round t across seeds
        cons_list = [rs[t]["consensus"] for rs in rounds_per_seed]
        cons_agg  = _agg_metrics(cons_list)

        # Scalar fields (elapsed, oracle_gap, sa_gain, etc.)
        scalar_keys = [k for k in rounds_per_seed[0][t]
                       if k not in ("consensus", "context_sizes", "elapsed")
                       and isinstance(rounds_per_seed[0][t][k], (int, float))]
        scalar_agg = {}
        for k in scalar_keys:
            vals = [rs[t][k] for rs in rounds_per_seed
                    if isinstance(rs[t].get(k), (int, float))]
            scalar_agg[k]         = float(np.mean(vals)) if vals else float("nan")
            scalar_agg[k+"_std"]  = float(np.std(vals, ddof=0)) if vals else float("nan")

        agg.append({
            "round":    t,
            "elapsed":  float(np.mean([rs[t].get("elapsed", 0)
                                       for rs in rounds_per_seed])),
            "consensus": cons_agg,
            **scalar_agg,
        })
    return agg


def aggregate_seeded_clf(seed_results: List[dict]) -> dict:
    """
    Merge n_seeds single-run clf result dicts into one aggregated dict.

    Every metric gets a companion _std key.  The structure is otherwise
    identical to a single run so all downstream table / figure code works
    without any changes — they just optionally display the _std values.
    """
    ref = seed_results[0]

    agg_rounds  = _agg_rounds([r["rounds"]       for r in seed_results])
    agg_sa      = _agg_metrics([r["single_agent"] for r in seed_results])
    agg_centr   = _agg_metrics([r["oracle"]       for r in seed_results])
    agg_fedavg  = _agg_metrics([r.get("fedavg", {}) for r in seed_results])

    return {
        # Condition identifiers — same for all seeds
        "dataset":   ref["dataset"],  "short":     ref["short"],
        "K":         ref["K"],        "partition": ref["partition"],
        "backbone":  ref["backbone"], "topology":  ref["topology"],
        "consensus": ref["consensus"],"tau":       ref["tau"],
        "alpha":     ref["alpha"],
        "n_seeds":   len(seed_results),
        "seeds_used":    [r.get("seed_used", i) for i,r in enumerate(seed_results)],
        # Aggregated metrics
        "rounds":       agg_rounds,
        "baseline":     agg_rounds[0],
        "final":        agg_rounds[-1],
        "single_agent": agg_sa,
        "oracle":       agg_centr,
        "fedavg":       agg_fedavg,
    }


def aggregate_seeded_reg(seed_results: List[dict]) -> dict:
    """Same as aggregate_seeded_clf but for regression result dicts."""
    ref = seed_results[0]

    agg_rounds  = _agg_rounds([r["rounds"]       for r in seed_results])
    agg_sa      = _agg_metrics([r["single_agent"] for r in seed_results])
    agg_centr   = _agg_metrics([r["oracle"]       for r in seed_results])
    agg_fedavg  = _agg_metrics([r.get("fedavg", {}) for r in seed_results])

    return {
        "dataset":   ref["dataset"],  "short":     ref["short"],
        "K":         ref["K"],        "partition": ref["partition"],
        "backbone":  ref["backbone"], "topology":  ref["topology"],
        "alpha":     ref["alpha"],
        "n_seeds":   len(seed_results),
        "seeds_used":    [r.get("seed_used", i) for i,r in enumerate(seed_results)],
        "rounds":       agg_rounds,
        "baseline":     agg_rounds[0],
        "final":        agg_rounds[-1],
        "single_agent": agg_sa,
        "oracle":       agg_centr,
        "fedavg":       agg_fedavg,
    }


def run_main_seeded(cfg, clf_cache, reg_cache):
    """
    Run all main experiment conditions over cfg.n_seeds seeds and return
    aggregated (mean ± std) result lists for classification and regression.

    For each (dataset, partition, K, backbone) condition:
      • Re-seed numpy + torch with each seed in cfg.seeds
      • Call run_dicl_clf / run_dicl_reg independently
      • Aggregate with aggregate_seeded_clf / aggregate_seeded_reg

    The datasets themselves are NOT re-split between seeds — the same
    train/test split is used (fixed by SEED at load time).  The random
    variation comes from:
        • Dirichlet partition draw  (agent shard assignment)
        • Query pool sampling
        • Pseudo-label selection (random subsampling within high-confidence set)
        • Reservoir sampling (context cap enforcement)
    This isolates algorithm stochasticity from data-split variance.
    """
    # ── Classification ────────────────────────────────────────────────────────
    clf_res = []
    conds = list(itertools.product(
        cfg.clf_datasets, ["iid","non_iid"], cfg.K_values, cfg.backbones))
    total = len(conds)
    for idx, (ds, part, K, bb) in enumerate(conds, 1):
        Xtr, Xte, ytr, yte, meta = clf_cache[ds]
        print(f"\n[CLF {idx:3d}/{total}] {meta['name'][:20]} | {part} | K={K} | {bb}")
        seed_runs = []
        for s, seed in enumerate(cfg.seeds):
            print(f"    seed {seed} ({s+1}/{cfg.n_seeds}) …", end=" ", flush=True)
            set_seed(seed)
            r = run_dicl_clf(Xtr, ytr, Xte, yte, K=K, partition=part,
                             backbone=bb, cfg=cfg, meta=meta, verbose=False)
            r["seed_used"] = seed
            seed_runs.append(r)
            dT = r["final"]["consensus"]["accuracy"]
            print(f"acc@T={dT:.4f}")
        clf_res.append(aggregate_seeded_clf(seed_runs))

    # ── Regression ────────────────────────────────────────────────────────────
    reg_res  = []
    K_reg    = 4
    conds_r  = list(itertools.product(
        cfg.reg_datasets, ["iid","non_iid"], cfg.backbones))
    total_r  = len(conds_r)
    for idx, (ds, part, bb) in enumerate(conds_r, 1):
        Xtr, Xte, ytr, yte, meta = reg_cache[ds]
        print(f"\n[REG {idx:3d}/{total_r}] {meta['name'][:20]} | {part} | {bb}")
        seed_runs = []
        for s, seed in enumerate(cfg.seeds):
            print(f"    seed {seed} ({s+1}/{cfg.n_seeds}) …", end=" ", flush=True)
            set_seed(seed)
            r = run_dicl_reg(Xtr, ytr, Xte, yte, K=K_reg, partition=part,
                             backbone=bb, cfg=cfg, meta=meta, verbose=False)
            r["seed_used"] = seed
            seed_runs.append(r)
            rmseT = r["final"]["consensus"]["rmse"]
            print(f"rmse@T={rmseT:.4f}")
        reg_res.append(aggregate_seeded_reg(seed_runs))

    return clf_res, reg_res


# =============================================================================
# Ablations
# =============================================================================

def _abl(cfg: Config, clf_cache: dict, **kwargs) -> list[RunResult]:
    """Helper: run both backbones on the ablation dataset with overrides."""
    Xtr, Xte, ytr, yte, meta = clf_cache[cfg.ablation_dataset]
    results = []
    for bb in cfg.backbones:
        r = run_dicl_clf(
            Xtr, ytr, Xte, yte,
            K=cfg.ablation_K, partition="non_iid",
            backbone=bb, cfg=cfg, meta=meta, verbose=True,
            **kwargs,
        )
        results.append(r)
    return results


def run_ablation_topology(cfg: Config, clf_cache: dict) -> list[RunResult]:
    results = []
    for topo in TOPOLOGIES:
        print(f"\n  [Topo] {topo}", end="")
        for r in _abl(cfg, clf_cache, topology=topo):
            r["ablation"] = "topology"
            results.append(r)
    return results


def run_ablation_consensus(cfg: Config, clf_cache: dict) -> list[RunResult]:
    results = []
    for cons in CONSENSUS_FNS:
        print(f"\n  [Cons] {cons}", end="")
        for r in _abl(cfg, clf_cache, consensus=cons):
            r["ablation"] = "consensus"
            results.append(r)
    return results


def run_ablation_tau(cfg: Config, clf_cache: dict) -> list[RunResult]:
    results = []
    for tau in [0.60, 0.70, 0.80, 0.90]:
        print(f"\n  [Tau] τ={tau}", end="")
        for r in _abl(cfg, clf_cache, tau=tau):
            r["ablation"] = "tau"
            results.append(r)
    return results


def run_ablation_K(cfg: Config, clf_cache: dict) -> list[RunResult]:
    Xtr, Xte, ytr, yte, meta = clf_cache[cfg.ablation_dataset]
    results = []
    for K in [2, 4, 8, 16]:
        print(f"\n  [K] K={K}", end="")
        for bb in cfg.backbones:
            r = run_dicl_clf(
                Xtr, ytr, Xte, yte, K=K, partition="iid",
                backbone=bb, cfg=cfg, meta=meta, verbose=True,
            )
            r["ablation"] = "K"
            results.append(r)
    return results


def run_ablation_alpha(cfg: Config, clf_cache: dict) -> list[RunResult]:
    results = []
    for alpha in [0.1, 0.5, 2.0]:
        print(f"\n  [Alpha] α={alpha}", end="")
        for r in _abl(cfg, clf_cache, alpha=alpha):
            r["ablation"] = "alpha"
            results.append(r)
    return results
