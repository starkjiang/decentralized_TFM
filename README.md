# D-ICL: Decentralised In-Context Learning for Tabular Foundation Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![TabICLv2](https://img.shields.io/badge/TabICLv2-ICML_2025-blueviolet.svg)](https://github.com/siyuanfan01/TabICL)
[![TabPFNv2](https://img.shields.io/badge/TabPFNv2-NeurIPS_2024-orange.svg)](https://github.com/automl/TabPFN)

Empirical validation of **D-ICL** — Decentralised In-Context Learning for
tabular foundation models — using **TabICLv2** (ICML 2025) and
**TabPFNv2** (NeurIPS 2024) as backbones.

D-ICL enables multiple agents, each holding a private shard of training data,
to collaboratively improve their in-context predictions without centralising raw data.
Agents share only predictions; raw features and labels never leave the local device.

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Experiments](#experiments)
  - [Main experiments](#main-experiments)
  - [Ablation studies](#ablation-studies)
- [Output Figures](#output-figures)
- [Design Decisions](#design-decisions)
- [Extending the Code](#extending-the-code)

---

## Overview

| Algorithm | Description |
|-----------|-------------|
| **Single-agent** | One TFM agent using only its initial local context C_k⁰ |
| **Centralised** | Single TFM with the full training set as context (oracle upper bound) |
| **D-ICL** | K agents with consensus + pseudo-label context enrichment |

**Key results:**
- D-ICL closes most of the gap between single-agent and centralised oracle.
- IID splits converge faster; non-IID splits converge to a higher residual
  gap that scales with the Dirichlet heterogeneity α.
- Consensus accuracy grows monotonically with the number of agents K
  (Proposition 1: variance ∝ 1/K for IID; +σ²_het for non-IID).
- The confidence threshold τ must exceed 0.5 for pseudo-labelling to be
  beneficial (Proposition 2).

---

## Method

D-ICL runs for T rounds. Each round has four phases:

```
Phase 1  Local ICL Inference
         Each agent k queries its backbone f_θ(x | C_k^t) on the test
         set and an unlabelled query pool Q ⊂ X_train.

Phase 2  (implicit) Neighbourhood topology
         Agents are connected by a graph A (fully-connected by default).

Phase 3  Consensus Aggregation
         Per-agent predictions are averaged over their neighbourhood:
           p̄_k^t(x) = (1/|N_k|) Σ_{j ∈ N_k} p_j^t(x)

Phase 4  Context Enrichment
         Pool points with consensus confidence ≥ τ are pseudo-labelled
         and appended to the agent's context reservoir C_k^{t+1}.
         Reservoir size is capped at m_max via random subsampling.
```

**Regression** uses inter-agent prediction standard deviation as a
confidence proxy: pool points with std ≤ median(std) are selected.

---

## Project Structure

```
dicl/
├── main.py                        # CLI entry point
├── requirements.txt
├── configs/
│   └── default.yaml               # All hyperparameters documented
├── dicl/                          # Core Python package
│   ├── __init__.py
│   ├── config.py                  # Config dataclass
│   ├── topology.py                # TOPOLOGIES, CONSENSUS_FNS, aggregate_all
│   ├── data.py                    # Dataset loaders + partitioning
│   ├── runner.py                  # Metrics, baselines, D-ICL loops, ablations
│   ├── reporting.py               # Console tables + JSON serialisation
│   ├── visualization.py           # 12 publication-quality figures
│   └── agents/
│       ├── __init__.py
│       ├── clf_agents.py          # TFMAgent, TabICLAgent, TabPFNAgent
│       └── reg_agents.py          # RegAgent, TabICLRegAgent, TabPFNRegAgent
├── figures/                       # (git-ignored) generated PDF + PNG figures
├── outputs/                       # (git-ignored) result JSONs
└── scripts/
    └── quick_test.sh              # 2-round smoke test
```

---

## Installation

### 1. Clone

```bash
git clone https://github.com/starkjiang/decentralized_TFM.git
cd dicl
```

### 2. Virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU** — For a T4 GPU (Google Colab), install the matching PyTorch wheel
> from [pytorch.org](https://pytorch.org/get-started/locally/) before
> running `pip install -r requirements.txt`. TabPFNv2 requires CUDA for
> reasonable runtime.

> **TabPFNv2** — Uses `ModelVersion.V2` (Apache-2.0), which does **not**
> require a HuggingFace login.

---

## Quick Start

### Full run (all datasets, all ablations)

```bash
python main.py
```

Results → `dicl_results.json`  
Figures → `./figures/`

### Smoke test (2 rounds, 1 dataset, 1 backbone)

```bash
bash scripts/quick_test.sh
# or:
python main.py --rounds 2 --clf-datasets breast_cancer \
               --reg-datasets diabetes_reg --backbones tabicl \
               --no-ablations
```

### All CLI options

```
usage: main.py [-h]
               [--clf-datasets ...]   classification datasets to run
               [--reg-datasets ...]   regression datasets to run
               [--backbones ...]      tabicl | tabpfn (default: both)
               [--rounds ROUNDS]      override T (D-ICL rounds)
               [--no-ablations]       skip ablation studies
               [--no-figures]         skip figure generation
               [--output OUTPUT]      results JSON path (default: dicl_results.json)
               [--seed SEED]          random seed (default: 42)
```

---

## Configuration

All hyperparameters live in two places that are kept in sync:

| Location | Purpose |
|----------|---------|
| `configs/default.yaml` | Human-readable reference |
| `dicl/config.py` — `Config` dataclass | Runtime source of truth |

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | `5` | D-ICL communication rounds |
| `m_0` | `64` | Initial context size per agent |
| `tau` | `0.80` | Pseudo-label confidence threshold |
| `delta_max` | `32` | Max pseudo-labels added per round |
| `m_max` | `1024` | Context reservoir capacity |
| `K_values` | `[2, 4, 8]` | Agent counts swept in main experiments |
| `alpha_dirichlet` | `0.5` | Dirichlet α for non-IID partition |
| `query_pool_frac` | `0.30` | Fraction of train set used as query pool |
| `backbones` | `["tabicl", "tabpfn"]` | TFM backbones to evaluate |

---

## Datasets

### Classification (5 datasets)

| Key | Dataset | Classes | Features |
|-----|---------|---------|---------|
| `breast_cancer` | Breast Cancer (UCI) | 2 | 30 |
| `wine` | Wine (UCI) | 3 | 13 |
| `iris` | Iris (Fisher) | 3 | 4 |
| `digits` | Digits (NIST) | 10 | 64 |
| `diabetes_clf` | Diabetes (binarised) | 2 | 10 |

### Regression (5 datasets)

| Key | Dataset | Features |
|-----|---------|---------|
| `california` | California Housing | 8 |
| `diabetes_reg` | Diabetes (UCI) | 10 |
| `linnerud` | Linnerud / Exercise | 3 |
| `energy` | Energy Efficiency (UCI OpenML) | 8 |
| `concrete` | Concrete Strength (UCI OpenML) | 8 |

All features are standardised with `StandardScaler`.
Regression targets are also standardised (scaler stored in `meta["y_scaler"]`).

---

## Experiments

### Main experiments

For each combination of (dataset × partition × K × backbone):

| Step | Description |
|------|-------------|
| Data split | Train/test 75/25 stratified split |
| Partitioning | IID (uniform random) and non-IID (Dirichlet, α=0.5) |
| Baselines | Single-agent (local context only) + Centralised oracle |
| D-ICL | T=5 rounds, fully-connected topology, arithmetic consensus |
| Metrics | Classification: Accuracy, F1-macro, ROC-AUC, log-loss |
| | Regression: RMSE, MAE, R² |

### Ablation studies

Five ablations, each run on `breast_cancer` with K=4, non-IID partition:

| Ablation | Values swept | What it tests |
|----------|-------------|---------------|
| **Topology** | fully_connected · ring · star · sparse_random | Effect of connectivity |
| **Consensus** | arithmetic · weighted · geometric | Aggregation strategy |
| **τ (tau)** | 0.60 · 0.70 · 0.80 · 0.90 | Pseudo-label threshold sensitivity |
| **K** | 2 · 4 · 8 · 16 | Agent count scaling |
| **α (alpha)** | 0.1 · 0.5 · 2.0 | Partition heterogeneity effect |

---

## Output Figures

After a full run, `./figures/` contains 12 figures (PDF + PNG, 300 DPI):

| File | Content |
|------|---------|
| `fig1_clf_convergence_iid` | Accuracy vs round for all clf datasets (IID, K=4) |
| `fig2_clf_convergence_noniid` | Same, non-IID |
| `fig3_clf_baseline_bar_iid` | Bar chart: SA vs D-ICL vs Centralised (IID) |
| `fig4_clf_baseline_bar_noniid` | Same, non-IID |
| `fig5_clf_k_scaling` | Accuracy vs K on breast_cancer |
| `fig6_clf_iid_vs_noniid` | IID vs non-IID convergence overlay (TabICLv2) |
| `fig7_theory_variance_reduction` | Proposition 1: Var[p̄] vs K |
| `fig8_theory_convergence_bound` | Proposition 3: loss gap bound vs T |
| `fig9_theory_tau_threshold` | Proposition 2: ε(τ) beneficial / harmful regions |
| `fig10_communication_cost` | Heatmap of communication cost vs K and pool size |
| `fig11_reg_convergence_iid` | RMSE vs round for all reg datasets (IID, K=4) |
| `fig12_reg_baseline_bar_iid` | Bar chart: SA vs D-ICL vs Centralised (regression) |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`_padded_context`** | Ensures all n_classes appear in context before each backbone fit, preventing silent NaN errors on skewed non-IID shards |
| **`_stratified_sample`** | Builds initial context with class balance even when the local shard is heavily imbalanced |
| **Variance-based selection for regression** | No natural confidence score exists for regressors; inter-agent std is a principled proxy for agreement |
| **Reservoir cap `m_max`** via random subsampling | Prevents unbounded memory growth while retaining diversity |
| **`ModelVersion.V2` for TabPFNv2** | Apache-2.0 licence; no HuggingFace authentication required |
| **Arithmetic mean as default consensus** | Equivalent to a product-of-Dirichlet posterior update under uniform priors; fastest to compute |

---

## Extending the Code

### Add a new classification dataset

Add a branch to `load_clf` in `dicl/data.py`:

```python
elif name == "my_dataset":
    X, y  = ...
    label = "My Dataset"
    cnames = [...]
```

Then add `"my_dataset"` to `cfg.clf_datasets`.

### Add a new graph topology

Add an entry to `TOPOLOGIES` in `dicl/topology.py`:

```python
def _topo_grid(K):
    side = int(K ** 0.5)
    A = np.zeros((K, K))
    ...
    return A

TOPOLOGIES["grid"] = _topo_grid
```

### Add a new backbone

Sub-class `TFMAgent` (classification) or `RegAgent` (regression) in
`dicl/agents/` and register it in the factory function:

```python
class MyAgent(TFMAgent):
    def _refresh_context(self): ...
    def _raw_predict_proba(self, X): ...

def make_clf_agent(backbone, agent_id, Xk, yk, n_cls, cfg):
    if backbone == "myagent": return MyAgent(...)
    ...
```

### Use a single function programmatically

```python
from dicl import Config, load_clf, run_dicl_clf

cfg              = Config(T=3, tau=0.85)
Xtr, Xte, ytr, yte, meta = load_clf("breast_cancer", cfg)
result = run_dicl_clf(Xtr, ytr, Xte, yte, K=4, partition="iid",
                      backbone="tabicl", cfg=cfg, meta=meta)
print(result["final"]["consensus"]["accuracy"])
```
