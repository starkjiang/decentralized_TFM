"""
dicl
=====
D-ICL: Decentralised In-Context Learning for Tabular Foundation Models.

Backbones : TabICLv2 (ICML 2025) + TabPFNv2 (NeurIPS 2024)
Tasks     : 5 Classification · 5 Regression
Baselines : Single-agent · Centralised (oracle) · D-ICL
Ablations : Topology · Consensus · Tau · K · Alpha

Package layout
--------------
config          — Config dataclass (all hyperparameters)
agents/         — Classification and regression agent classes
  clf_agents    — TFMAgent, TabICLAgent, TabPFNAgent, make_clf_agent
  reg_agents    — RegAgent, TabICLRegAgent, TabPFNRegAgent, make_reg_agent
topology        — TOPOLOGIES, CONSENSUS_FNS, aggregate_all
data            — Dataset loaders and IID / non-IID partitioning
runner          — Metrics, baselines, D-ICL loops, ablations
reporting       — Console tables and JSON serialisation
visualization   — 12 publication-quality figure generators
"""

from .config        import Config
from .agents        import (
    TFMAgent, TabICLAgent, TabPFNAgent, make_clf_agent,
    RegAgent, TabICLRegAgent, TabPFNRegAgent, make_reg_agent,
)
from .topology      import TOPOLOGIES, CONSENSUS_FNS, aggregate_all
from .data          import load_clf, load_reg, partition_iid, partition_noniid
from .runner        import (
    eval_clf, eval_reg,
    single_agent_clf, centralised_clf,
    single_agent_reg, centralised_reg,
    run_dicl_clf, run_dicl_reg,
    run_main,
    run_ablation_topology, run_ablation_consensus,
    run_ablation_tau, run_ablation_K, run_ablation_alpha,
)
from .reporting     import (
    print_clf_table, print_reg_table, print_ablation_table,
    save_results_json,
)
from .visualization import build_figures

__all__ = [
    "Config",
    "TFMAgent", "TabICLAgent", "TabPFNAgent", "make_clf_agent",
    "RegAgent", "TabICLRegAgent", "TabPFNRegAgent", "make_reg_agent",
    "TOPOLOGIES", "CONSENSUS_FNS", "aggregate_all",
    "load_clf", "load_reg", "partition_iid", "partition_noniid",
    "eval_clf", "eval_reg",
    "single_agent_clf", "centralised_clf",
    "single_agent_reg", "centralised_reg",
    "run_dicl_clf", "run_dicl_reg",
    "run_main",
    "run_ablation_topology", "run_ablation_consensus",
    "run_ablation_tau", "run_ablation_K", "run_ablation_alpha",
    "print_clf_table", "print_reg_table", "print_ablation_table",
    "save_results_json",
    "build_figures",
]
