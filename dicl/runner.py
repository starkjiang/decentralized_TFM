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
from typing import Any

import numpy as np
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
