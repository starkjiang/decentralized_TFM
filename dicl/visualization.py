"""
dicl/visualization.py
======================
Publication-quality figure generators (300 DPI, PDF + PNG).

Figure inventory (12 total)
---------------------------
  fig1  — Classification convergence (IID, K=4)
  fig2  — Classification convergence (non-IID, K=4)
  fig3  — Classification baseline bar chart (IID)
  fig4  — Classification baseline bar chart (non-IID)
  fig5  — Effect of K on accuracy
  fig6  — IID vs non-IID convergence (TabICLv2, K=4, first 3 datasets)
  fig7  — Theory: variance bound vs K (Proposition 1)
  fig8  — Theory: convergence bound vs T (Proposition 3)
  fig9  — Theory: τ threshold effect (Proposition 2)
  fig10 — Communication cost heatmap
  fig11 — Regression convergence (IID, K=4)
  fig12 — Regression baseline bar chart (IID)

All saved to: ``./figures/``
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config

# ── Global typography (NeurIPS / ICML camera-ready style) ────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          13,
    "axes.titlesize":     13,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "legend.framealpha":  0.85,
    "lines.linewidth":    2.0,
    "lines.markersize":   6,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── Colour palettes ───────────────────────────────────────────────────────────
K_CLR   = {2: "#2E5FA3", 4: "#E05C1A", 8: "#2DA34D", 16: "#8E44AD"}
BB_CLR  = {"tabicl": "#2E5FA3", "tabpfn": "#E05C1A"}
DS_CLR  = ["#2E5FA3", "#E05C1A", "#2DA34D", "#8E44AD", "#C0392B"]

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ── I/O helper ────────────────────────────────────────────────────────────────
def _save(fig, name: str) -> None:
    """Save figure as PDF (vector) and PNG (raster preview) at 300 DPI."""
    base = os.path.join(FIG_DIR, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    print(f"  → {base}.pdf / .png")


# =============================================================================
# Classification figures
# =============================================================================

def _clf_convergence_iid(clf_res, datasets, ds_names):
    """Fig 1 — Classification convergence (IID, K=4, all datasets)."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for di, (ds, ax) in enumerate(zip(datasets, axes)):
        for bb, ls in [("tabicl", "-"), ("tabpfn", "--")]:
            sub = [r for r in clf_res if r["short"] == ds and r["K"] == 4
                   and r["partition"] == "iid" and r["backbone"] == bb]
            if not sub:
                continue
            xs = [r["round"] for r in sub[0]["rounds"]]
            ys = [r["consensus"]["accuracy"] for r in sub[0]["rounds"]]
            ax.plot(xs, ys, ls=ls, marker="o", color=BB_CLR[bb],
                    label=f"D-ICL ({bb[:5]})")
            ax.axhline(sub[0]["oracle"]["accuracy"],
                       color=BB_CLR[bb], ls=":", lw=1.4, alpha=0.7,
                       label=f"Centralised ({bb[:5]})")
            ax.axhline(sub[0]["single_agent"]["accuracy"],
                       color=BB_CLR[bb], ls="-.", lw=1.1, alpha=0.55,
                       label=f"Single-agent ({bb[:5]})")
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.3, 1.05)
        ax.legend(ncol=1, loc="lower right")
        ax.set_xticks(xs)
        ax.set_title(f"{ds_names.get(ds, ds)[:18]}", fontsize=9)

    fig.subplots_adjust(bottom=0.22)
    return fig


def _clf_convergence_noniid(clf_res, datasets, ds_names):
    """Fig 2 — Classification convergence (non-IID, K=4)."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ds, ax in zip(datasets, axes):
        for bb, ls in [("tabicl", "-"), ("tabpfn", "--")]:
            sub = [r for r in clf_res if r["short"] == ds and r["K"] == 4
                   and r["partition"] == "non_iid" and r["backbone"] == bb]
            if not sub:
                continue
            xs = [r["round"] for r in sub[0]["rounds"]]
            ys = [r["consensus"]["accuracy"] for r in sub[0]["rounds"]]
            ax.plot(xs, ys, ls=ls, marker="s", color=BB_CLR[bb],
                    label=f"D-ICL ({bb[:5]})")
            ax.axhline(sub[0]["oracle"]["accuracy"],
                       color=BB_CLR[bb], ls=":", lw=1.4, alpha=0.7)
            ax.axhline(sub[0]["single_agent"]["accuracy"],
                       color=BB_CLR[bb], ls="-.", lw=1.1, alpha=0.55)
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.3, 1.05)
        ax.legend(loc="lower right")
        ax.set_xticks(xs)
        ax.set_title(f"{ds_names.get(ds, ds)}", fontsize=9)

    fig.subplots_adjust(bottom=0.22)
    return fig


def _clf_baseline_bar(clf_res, datasets, partition):
    """Fig 3 / 4 — Classification baseline comparison bar chart."""
    lbls, sa_, icl_, pfn_, centr_ = [], [], [], [], []
    for ds in datasets:
        si = [r for r in clf_res if r["short"] == ds and r["K"] == 4
              and r["partition"] == partition and r["backbone"] == "tabicl"]
        sp = [r for r in clf_res if r["short"] == ds and r["K"] == 4
              and r["partition"] == partition and r["backbone"] == "tabpfn"]
        if not si:
            continue
        lbls.append(si[0]["dataset"].split("(")[0].strip())
        sa_.append(si[0]["single_agent"]["accuracy"])
        icl_.append(si[0]["final"]["consensus"]["accuracy"])
        pfn_.append(sp[0]["final"]["consensus"]["accuracy"] if sp else 0)
        centr_.append(si[0]["oracle"]["accuracy"])

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(lbls)), 4.5))
    x = np.arange(len(lbls))
    w = 0.20
    ax.bar(x - 1.5 * w, sa_,    w, alpha=0.55, color="#888888",        label="Single-agent")
    ax.bar(x - 0.5 * w, icl_,   w, alpha=0.90, color=BB_CLR["tabicl"], label="D-ICL / TabICLv2")
    ax.bar(x + 0.5 * w, pfn_,   w, alpha=0.90, color=BB_CLR["tabpfn"], label="D-ICL / TabPFNv2")
    ax.bar(x + 1.5 * w, centr_, w, alpha=0.55, color="#2DA34D",         label="Centralised")
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.18)
    ax.legend(ncol=2)
    return fig


def _clf_k_scaling(clf_res, cfg: Config):
    """Fig 5 — Effect of number of agents K on accuracy."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for bb, ls in [("tabicl", "-"), ("tabpfn", "--")]:
        accs = []
        for K in cfg.K_values:
            sub = [r for r in clf_res if r["short"] == "breast_cancer" and r["K"] == K
                   and r["partition"] == "iid" and r["backbone"] == bb]
            accs.append(sub[0]["final"]["consensus"]["accuracy"] if sub else float("nan"))
        ax.plot(cfg.K_values, accs, ls=ls, marker="o",
                color=BB_CLR[bb], label=f"D-ICL / {bb[:5]}")

    ref = [r for r in clf_res if r["short"] == "breast_cancer" and r["K"] == 4
           and r["partition"] == "iid" and r["backbone"] == "tabicl"]
    if ref:
        ax.axhline(ref[0]["single_agent"]["accuracy"],
                   color="#888", ls="-.", lw=1.4, label="Single-agent")
        ax.axhline(ref[0]["oracle"]["accuracy"],
                   color="#2DA34D", ls=":", lw=1.4, label="Centralised")
    ax.set_xlabel("Number of agents $K$")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(cfg.K_values)
    ax.legend()
    return fig


def _clf_iid_vs_noniid(clf_res, datasets):
    """Fig 6 — IID vs Non-IID convergence (TabICLv2, K=4, first 3 datasets)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)
    for di, (ds, ax) in enumerate(zip(datasets[:3], axes)):
        for pt, ls in [("iid", "-"), ("non_iid", "--")]:
            sub = [r for r in clf_res if r["short"] == ds and r["K"] == 4
                   and r["partition"] == pt and r["backbone"] == "tabicl"]
            if not sub:
                continue
            xs  = [r["round"] for r in sub[0]["rounds"]]
            ys  = [r["consensus"]["accuracy"] for r in sub[0]["rounds"]]
            lbl = "IID" if pt == "iid" else "Non-IID"
            ax.plot(xs, ys, ls=ls, marker="o", color=DS_CLR[di], label=lbl)
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_xticks(xs)
        name = sub[0]["dataset"] if sub else ds
        ax.set_title(f"{name.split('(')[0].strip()}", fontsize=9)

    fig.subplots_adjust(bottom=0.25)
    return fig


# =============================================================================
# Theory figures
# =============================================================================

def _theory_variance(cfg: Config):
    """Fig 7 — Proposition 1: variance bound vs K."""
    Kr  = np.arange(1, 17)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.semilogy(Kr, cfg.sigma_base ** 2 / Kr,
                "o-", color="#2E5FA3", label="IID")
    ax.semilogy(Kr, cfg.sigma_base ** 2 / Kr + cfg.sigma_het_noniid ** 2,
                "s--", color="#C0392B", label="Non-IID")
    ax.fill_between(Kr,
                    cfg.sigma_base ** 2 / Kr,
                    cfg.sigma_base ** 2 / Kr + cfg.sigma_het_noniid ** 2,
                    alpha=0.12, color="#C0392B")
    ax.set_xlabel("Number of agents $K$")
    ax.set_ylabel(r"$\mathrm{Var}[\bar{p}^t]$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return fig


def _theory_convergence(cfg: Config):
    """Fig 8 — Proposition 3: convergence bound vs T."""
    Tr = np.arange(1, 21)
    C1 = cfg.L_lipschitz * 2
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for K, col in [(2, K_CLR[2]), (4, K_CLR[4]), (8, K_CLR[8])]:
        ax.plot(Tr, C1 / np.sqrt(Tr),
                "-", color=col, label=f"$K={K}$, IID")
        ax.plot(Tr, C1 / np.sqrt(Tr) + cfg.sigma_het_noniid / np.sqrt(K),
                "--", color=col, alpha=0.65, label=f"$K={K}$, non-IID")
    ax.set_xlabel("Rounds $T$")
    ax.set_ylabel("Loss gap bound")
    ax.legend(ncol=2, fontsize=10)
    return fig


def _theory_tau(cfg: Config):
    """Fig 9 — Proposition 2: ε(τ) threshold effect."""
    tau_r = np.linspace(0.5, 0.99, 120)
    eps   = -(2 * tau_r - 1) * 0.15
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(tau_r, eps, color="#2DA34D", lw=2.2)
    ax.axhline(0, color="black", ls="--", lw=1.0, alpha=0.6)
    ax.fill_between(tau_r, eps, 0, where=(eps < 0),
                    alpha=0.18, color="#2DA34D",
                    label=r"$\varepsilon(\kappa)<0$ (beneficial)")
    ax.fill_between(tau_r, eps, 0, where=(eps >= 0),
                    alpha=0.14, color="#C0392B",
                    label=r"$\varepsilon(\kappa)\geq0$ (harmful)")
    ax.axvline(cfg.tau, color="#E05C1A", ls=":", lw=2.0,
               label=rf"$\tau={cfg.tau}$ (default)")
    ax.set_xlabel(r"Confidence threshold $\tau$")
    ax.set_ylabel(r"$\varepsilon(\tau)$")
    ax.legend()
    return fig


def _comm_cost_heatmap(cfg: Config):
    """Fig 10 — Communication cost heatmap (K × query pool size)."""
    Kv   = [2, 4, 8, 16]
    Mv   = [50, 100, 200, 500]
    cost = np.array([[K * M * 10 * cfg.T for M in Mv] for K in Kv])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sns.heatmap(
        np.log10(cost + 1), ax=ax,
        xticklabels=[f"$M={m}$" for m in Mv],
        yticklabels=[f"$K={k}$" for k in Kv],
        annot=[[f"{v / 1000:.1f}k" for v in row] for row in cost],
        fmt="", cmap="Blues", linewidths=0.5,
        cbar_kws={"label": r"$\log_{10}(\mathrm{floats})$", "shrink": 0.85},
    )
    ax.set_xlabel("Query pool size $|\\mathcal{Q}|$")
    ax.set_ylabel("Number of agents $K$")
    return fig


# =============================================================================
# Regression figures
# =============================================================================

def _reg_convergence_iid(reg_res, reg_ds):
    """Fig 11 — Regression convergence (IID, K=4, all datasets)."""
    n    = len(reg_ds)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ds, ax in zip(reg_ds, axes):
        for bb, ls in [("tabicl", "-"), ("tabpfn", "--")]:
            sub = [r for r in reg_res if r["short"] == ds
                   and r["partition"] == "iid" and r["backbone"] == bb]
            if not sub:
                continue
            xs = [r["round"] for r in sub[0]["rounds"]]
            ys = [r["consensus"]["rmse"] for r in sub[0]["rounds"]]
            ax.plot(xs, ys, ls=ls, marker="o", color=BB_CLR[bb],
                    label=f"D-ICL ({bb[:5]})")
            ax.axhline(sub[0]["oracle"]["rmse"],
                       color=BB_CLR[bb], ls=":", lw=1.4, alpha=0.7,
                       label=f"Centralised ({bb[:5]})")
            ax.axhline(sub[0]["single_agent"]["rmse"],
                       color=BB_CLR[bb], ls="-.", lw=1.1, alpha=0.55,
                       label=f"Single-agent ({bb[:5]})")
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("RMSE")
        ax.legend(ncol=1, loc="upper right")
        ax.set_xticks(xs)
        ax.set_title(f"{sub[0]['dataset'] if sub else ds}", fontsize=9)

    fig.subplots_adjust(bottom=0.22)
    return fig


def _reg_baseline_bar(reg_res, reg_ds, partition):
    """Fig 12 — Regression baseline comparison bar chart."""
    lbls, sa_, dicl_, centr_ = [], [], [], []
    for ds in reg_ds:
        for bb in ["tabicl", "tabpfn"]:
            sub = [r for r in reg_res if r["short"] == ds
                   and r["partition"] == partition and r["backbone"] == bb]
            if not sub:
                continue
            lbls.append(
                f"{sub[0]['dataset'].split('(')[0].strip()[:12]}\n({bb[:5]})"
            )
            sa_.append(sub[0]["single_agent"]["rmse"])
            dicl_.append(sub[0]["final"]["consensus"]["rmse"])
            centr_.append(sub[0]["oracle"]["rmse"])

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(lbls)), 4.8))
    x = np.arange(len(lbls))
    w = 0.27
    ax.bar(x - w, sa_,    w, alpha=0.60, color="#888888", label="Single-agent")
    ax.bar(x,     dicl_,  w, alpha=0.90, color="#2E5FA3", label="D-ICL @ $T$")
    ax.bar(x + w, centr_, w, alpha=0.60, color="#2DA34D", label="Centralised")
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("RMSE (lower is better)")
    ax.legend()
    fig.subplots_adjust(bottom=0.28)
    return fig


# =============================================================================
# Orchestrator
# =============================================================================

def build_figures(clf_res: list, reg_res: list, cfg: Config) -> None:
    """Save all 12 figures to ``./figures/`` as PDF + PNG at 300 DPI."""
    datasets = list(dict.fromkeys(r["short"] for r in clf_res))
    ds_names = {r["short"]: r["dataset"] for r in clf_res}
    reg_ds   = list(dict.fromkeys(r["short"] for r in reg_res))

    sns.set_style("whitegrid")

    figs = [
        (_clf_convergence_iid(clf_res, datasets, ds_names),  "fig1_clf_convergence_iid"),
        (_clf_convergence_noniid(clf_res, datasets, ds_names),"fig2_clf_convergence_noniid"),
        (_clf_baseline_bar(clf_res, datasets, "iid"),         "fig3_clf_baseline_bar_iid"),
        (_clf_baseline_bar(clf_res, datasets, "non_iid"),     "fig4_clf_baseline_bar_noniid"),
        (_clf_k_scaling(clf_res, cfg),                        "fig5_clf_k_scaling"),
        (_clf_iid_vs_noniid(clf_res, datasets),               "fig6_clf_iid_vs_noniid"),
        (_theory_variance(cfg),                               "fig7_theory_variance_reduction"),
        (_theory_convergence(cfg),                            "fig8_theory_convergence_bound"),
        (_theory_tau(cfg),                                    "fig9_theory_tau_threshold"),
        (_comm_cost_heatmap(cfg),                             "fig10_communication_cost"),
        (_reg_convergence_iid(reg_res, reg_ds),               "fig11_reg_convergence_iid"),
        (_reg_baseline_bar(reg_res, reg_ds, "iid"),           "fig12_reg_baseline_bar_iid"),
    ]

    for fig, name in figs:
        _save(fig, name)

    print(f"\n  All {len(figs)} figures saved to ./{FIG_DIR}/")
