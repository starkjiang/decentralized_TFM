"""
dicl/config.py
==============
Central configuration for D-ICL experiments.

All hyperparameters, dataset lists, and ablation flags live here.
The ``Config`` dataclass is the single source of truth passed to every
experiment function — no global state leaks into the rest of the codebase.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Datasets ──────────────────────────────────────────────────────────────
    clf_datasets: List[str] = field(default_factory=lambda: [
        "breast_cancer", "wine", "iris", "digits", "diabetes_clf",
    ])
    reg_datasets: List[str] = field(default_factory=lambda: [
        "california", "diabetes_reg", "linnerud", "energy", "concrete",
    ])
    backbones: List[str] = field(default_factory=lambda: ["tabicl", "tabpfn"])

    # ── D-ICL hyperparameters ─────────────────────────────────────────────────
    K_values:        List[int] = field(default_factory=lambda: [2, 4, 8])
    T:               int   = 5       # number of D-ICL rounds
    m_0:             int   = 64      # initial context size per agent
    tau:             float = 0.80    # pseudo-label confidence threshold
    delta_max:       int   = 32      # max pseudo-labels added per round
    m_max:           int   = 1024    # maximum context reservoir size
    alpha_dirichlet: float = 0.5     # Dirichlet concentration for non-IID split
    test_size:       float = 0.25    # train/test split fraction
    query_pool_frac: float = 0.30    # fraction of training data used as query pool

    # ── TabICLv2 backbone ─────────────────────────────────────────────────────
    tabicl_n_estimators:   int   = 16
    tabicl_batch_size:     int   = 8
    tabicl_kv_cache:       bool  = True
    tabicl_avg_logits:     bool  = True
    tabicl_temperature:    float = 0.9
    tabicl_clf_checkpoint: str   = "tabicl-classifier-v2-20260212.ckpt"
    tabicl_reg_checkpoint: str   = "tabicl-regressor-v2-20260212.ckpt"

    # ── Theory parameters (used in analytical figures) ────────────────────────
    sigma_base:       float = 0.10
    sigma_het_noniid: float = 0.08
    L_lipschitz:      float = 1.00

    # ── Ablation flags ────────────────────────────────────────────────────────
    run_ablation_topology:  bool = True
    run_ablation_consensus: bool = True
    run_ablation_tau:       bool = True
    run_ablation_K:         bool = True
    run_ablation_alpha:     bool = True
    ablation_dataset:       str  = "breast_cancer"
    ablation_K:             int  = 4
