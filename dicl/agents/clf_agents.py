"""
dicl/agents/clf_agents.py
==========================
Classification agents for D-ICL.

Class hierarchy
---------------
  TFMAgent (ABC)
    ├── TabICLAgent   — wraps TabICLv2 classifier
    └── TabPFNAgent   — wraps TabPFNv2 classifier (ModelVersion.V2, Apache-2.0)

Design decisions
----------------
- ``_padded_context`` ensures all n_classes appear in the context before
  each fit(), preventing silent NaN errors on non-IID splits.
- ``_stratified_sample`` builds the initial context so that every class
  is represented even when local data is severely imbalanced.
- ``predict_proba`` re-maps classifier output columns back to the global
  class ordering so aggregation across agents is dimension-consistent.
- ``update_context`` applies the τ-threshold pseudo-label rule and caps
  the reservoir at m_max via random subsampling.
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

import torch
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Abstract base
# =============================================================================

class TFMAgent(ABC):
    """
    Abstract classification agent.

    Sub-classes must implement:
      _refresh_context() — refit the backbone on (C_x, C_y)
      _raw_predict_proba(X) — return raw probability array from backbone
      backbone (property) — string label used in result dicts
    """

    def __init__(self, agent_id, X_local, y_local, n_classes, cfg):
        self.id        = agent_id
        self.X_local   = X_local
        self.y_local   = y_local
        self.n_classes = n_classes
        self.cfg       = cfg

        n_ctx   = min(cfg.m_0, len(X_local))
        ctx_idx = self._stratified_sample(n_ctx)

        self.C_x = X_local[ctx_idx].copy()
        self.C_y = y_local[ctx_idx].copy()

        mask          = np.ones(len(X_local), bool)
        mask[ctx_idx] = False
        self.X_unused = X_local[mask]
        self.y_unused = y_local[mask]

        self._fitted_classes = np.arange(n_classes)
        self._refresh_context()

    # ── Initialisation helpers ─────────────────────────────────────────────

    def _stratified_sample(self, n: int) -> np.ndarray:
        """Return indices for a stratified random sample of size n."""
        classes = np.unique(self.y_local)
        idx     = []
        per_cls = max(1, n // len(classes))

        for c in classes:
            pool = np.where(self.y_local == c)[0]
            idx.extend(
                np.random.choice(pool, min(per_cls, len(pool)), replace=False).tolist()
            )

        idx = list(set(idx))
        rem = [i for i in range(len(self.y_local)) if i not in set(idx)]
        while len(idx) < n and rem:
            pick = int(np.random.choice(rem))
            idx.append(pick)
            rem.remove(pick)
        return np.array(idx[:n])

    def _padded_context(self):
        """
        Return (Cx, Cy) with at least one example per class.

        Draws missing-class examples first from X_unused, then from
        X_local as a fallback, so the backbone never crashes on a
        single-class context.
        """
        Cx, Cy = self.C_x.copy(), self.C_y.copy()
        for c in range(self.n_classes):
            if c not in Cy:
                pool = np.where(self.y_unused == c)[0]
                if len(pool):
                    Cx = np.vstack([Cx, self.X_unused[pool[0]:pool[0] + 1]])
                    Cy = np.append(Cy, c)
                else:
                    pool2 = np.where(self.y_local == c)[0]
                    if len(pool2):
                        Cx = np.vstack([Cx, self.X_local[pool2[0]:pool2[0] + 1]])
                        Cy = np.append(Cy, c)
        return Cx, Cy

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def _refresh_context(self) -> None: ...

    @abstractmethod
    def _raw_predict_proba(self, X) -> np.ndarray: ...

    @property
    @abstractmethod
    def backbone(self) -> str: ...

    # ── Public API ─────────────────────────────────────────────────────────

    def predict_proba(self, X_query: np.ndarray) -> np.ndarray:
        """
        Return (N, n_classes) probability array aligned to the global
        class ordering.
        """
        raw = self._raw_predict_proba(X_query)
        if raw.shape[1] != self.n_classes:
            proba = np.full((len(X_query), self.n_classes), 1e-9)
            for i, c in enumerate(self._fitted_classes):
                proba[:, int(c)] = raw[:, i]
        else:
            proba = raw
        proba = np.clip(proba, 1e-9, 1.0)
        proba /= proba.sum(1, keepdims=True)
        return proba

    def update_context(
        self,
        X_pool: np.ndarray,
        consensus_proba: np.ndarray,
    ) -> int:
        """
        Append high-confidence pseudo-labelled pool points to the context.

        Parameters
        ----------
        X_pool          : unlabelled query pool
        consensus_proba : (N_pool, n_classes) neighbourhood-aggregated probabilities

        Returns
        -------
        Number of pseudo-labels added.
        """
        conf     = consensus_proba.max(1)
        pseudo_y = consensus_proba.argmax(1).astype(int)
        high     = np.where(conf >= self.cfg.tau)[0]

        if not len(high):
            return 0

        n_add  = min(len(high), self.cfg.delta_max)
        chosen = np.random.choice(high, n_add, replace=False)

        self.C_x = np.vstack([self.C_x, X_pool[chosen]])
        self.C_y = np.append(self.C_y, pseudo_y[chosen])

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

class TabICLAgent(TFMAgent):
    """D-ICL agent backed by TabICLv2 classifier."""

    def __init__(self, agent_id, X_local, y_local, n_classes, cfg):
        self.clf = TabICLClassifier(
            n_estimators      = cfg.tabicl_n_estimators,
            batch_size        = cfg.tabicl_batch_size,
            kv_cache          = cfg.tabicl_kv_cache,
            average_logits    = cfg.tabicl_avg_logits,
            softmax_temperature = cfg.tabicl_temperature,
            checkpoint_version  = cfg.tabicl_clf_checkpoint,
            device             = None,
            allow_auto_download = True,
        )
        super().__init__(agent_id, X_local, y_local, n_classes, cfg)

    def _refresh_context(self):
        Cx, Cy = self._padded_context()
        self.clf.fit(Cx, Cy)
        self._fitted_classes = np.unique(Cy)

    def _raw_predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)

    @property
    def backbone(self) -> str:
        return "TabICLv2"


class TabPFNAgent(TFMAgent):
    """
    D-ICL agent backed by TabPFNv2 (ModelVersion.V2, Apache-2.0).

    Uses a concrete device string so the classifier does not attempt
    an automatic device discovery that can silently fall back to CPU.
    """

    def __init__(self, agent_id, X_local, y_local, n_classes, cfg):
        self.clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device=DEVICE
        )
        super().__init__(agent_id, X_local, y_local, n_classes, cfg)

    def _refresh_context(self):
        Cx, Cy = self._padded_context()
        self.clf.fit(Cx, Cy)
        self._fitted_classes = np.unique(Cy)

    def _raw_predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)

    @property
    def backbone(self) -> str:
        return "TabPFNv2"


# =============================================================================
# Factory
# =============================================================================

def make_clf_agent(backbone: str, agent_id, Xk, yk, n_cls, cfg) -> TFMAgent:
    """Instantiate a classification agent by backbone name."""
    if backbone == "tabicl":
        return TabICLAgent(agent_id, Xk, yk, n_cls, cfg)
    elif backbone == "tabpfn":
        return TabPFNAgent(agent_id, Xk, yk, n_cls, cfg)
    raise ValueError(f"Unknown backbone: {backbone!r}")
