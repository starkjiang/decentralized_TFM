"""
dicl/agents/reg_agents.py
==========================
Regression agents for D-ICL.

Class hierarchy
---------------
  RegAgent (ABC)
    ├── TabICLRegAgent  — wraps TabICLv2 native regressor
    └── TabPFNRegAgent  — wraps TabPFNv2 native regressor

Design decisions
----------------
- ``update_context_reg`` uses inter-agent prediction variance as a
  confidence proxy: pool points where std(predictions) is below the
  median threshold are selected as pseudo-labels.
- The context reservoir is capped at m_max via random subsampling,
  mirroring the classification agent behaviour.
- Both agents standardise targets in the data loader, so all
  predictions are on the same scale.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Abstract base
# =============================================================================

class RegAgent(ABC):
    """
    Abstract regression agent with reservoir context management.

    Sub-classes must implement:
      _refresh_context() — refit the backbone on (C_x, C_y)
      predict(X)         — return float predictions
      backbone (property) — string label used in result dicts
    """

    def __init__(self, agent_id, X_local, y_local, cfg):
        self.id      = agent_id
        self.X_local = X_local
        self.y_local = y_local.astype(float)
        self.cfg     = cfg

        n_ctx = min(cfg.m_0, len(X_local))
        idx   = np.random.choice(len(X_local), n_ctx, replace=False)

        self.C_x = X_local[idx].copy()
        self.C_y = self.y_local[idx].copy()

        mask          = np.ones(len(X_local), bool)
        mask[idx]     = False
        self.X_unused = X_local[mask]
        self.y_unused = self.y_local[mask]

        self._refresh_context()

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def _refresh_context(self) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def backbone(self) -> str: ...

    # ── Public API ─────────────────────────────────────────────────────────

    def update_context_reg(
        self,
        X_pool:            np.ndarray,
        consensus_preds:   np.ndarray,
        agent_preds_list:  list[np.ndarray],
    ) -> int:
        """
        Append low-variance pseudo-labelled pool points to the context.

        Uses inter-agent prediction std as a confidence proxy: points
        where std ≤ median(std) are considered "agreed upon" and added.

        Parameters
        ----------
        X_pool           : unlabelled query pool
        consensus_preds  : neighbourhood-averaged predictions for pool
        agent_preds_list : list of per-agent raw predictions for pool

        Returns
        -------
        Number of pseudo-labels added.
        """
        stds   = np.std(np.stack(agent_preds_list, 0), 0)
        thresh = np.percentile(stds, 50)
        high   = np.where(stds <= thresh)[0]

        if not len(high):
            return 0

        n_add  = min(len(high), self.cfg.delta_max)
        chosen = np.random.choice(high, n_add, replace=False)

        self.C_x = np.vstack([self.C_x, X_pool[chosen]])
        self.C_y = np.append(self.C_y, consensus_preds[chosen])

        if len(self.C_y) > self.cfg.m_max:
            keep     = np.random.choice(len(self.C_y), self.cfg.m_max, replace=False)
            self.C_x = self.C_x[keep]
            self.C_y = self.C_y[keep]

        self._refresh_context()
        return n_add

    @property
    def context_size(self) -> int:
        return int(len(self.C_y))


# =============================================================================
# Concrete agents
# =============================================================================

class TabICLRegAgent(RegAgent):
    """D-ICL regression agent backed by TabICLv2 native regressor."""

    def __init__(self, agent_id, X_local, y_local, cfg):
        self.clf = TabICLRegressor(
            n_estimators       = cfg.tabicl_n_estimators,
            batch_size         = cfg.tabicl_batch_size,
            kv_cache           = cfg.tabicl_kv_cache,
            checkpoint_version = cfg.tabicl_reg_checkpoint,
            device             = None,
            allow_auto_download = True,
        )
        super().__init__(agent_id, X_local, y_local, cfg)

    def _refresh_context(self):
        self.clf.fit(self.C_x, self.C_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X).astype(float)

    @property
    def backbone(self) -> str:
        return "TabICLv2"


class TabPFNRegAgent(RegAgent):
    """D-ICL regression agent backed by TabPFNv2 native regressor."""

    def __init__(self, agent_id, X_local, y_local, cfg):
        self.model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device=DEVICE
        )
        super().__init__(agent_id, X_local, y_local, cfg)

    def _refresh_context(self):
        self.model.fit(self.C_x, self.C_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(float)

    @property
    def backbone(self) -> str:
        return "TabPFNv2"


# =============================================================================
# Factory
# =============================================================================

def make_reg_agent(backbone: str, agent_id, Xk, yk, cfg) -> RegAgent:
    """Instantiate a regression agent by backbone name."""
    if backbone == "tabicl":
        return TabICLRegAgent(agent_id, Xk, yk, cfg)
    elif backbone == "tabpfn":
        return TabPFNRegAgent(agent_id, Xk, yk, cfg)
    raise ValueError(f"Unknown backbone: {backbone!r}")
