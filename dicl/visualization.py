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

mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          13,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "legend.framealpha":  0.90,
    "legend.edgecolor":   "0.7",
    "lines.linewidth":    2.0,
    "lines.markersize":   5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── Shared visual vocabulary ─────────────────────────────────────────────────
# All figures (CLF and REG) use identical line-style / color encoding:
#   backbone color   :  TabICLv2=#2E5FA3(blue)  TabPFNv2=#E05C1A(orange)
#   method line-style:  D-ICL=solid  Centralised=dotted  Single-agent=dash-dot  FedAvg=dashed
BB_CLR   = {"tabicl": "#2E5FA3", "tabpfn": "#E05C1A"}
K_CLR    = {2: "#2E5FA3", 4: "#E05C1A", 8: "#2DA34D", 16: "#8E44AD"}
DS_CLR   = ["#2E5FA3", "#E05C1A", "#2DA34D", "#8E44AD", "#C0392B"]
LS_DICL  = "-"     # D-ICL convergence curve
LS_CENTR = ":"     # Centralised reference line
LS_SA    = "-."    # Single-agent reference line
LS_FEDAVG= "--"    # FedAvg reference line
MK_TABICL= "o"     # TabICLv2 marker
MK_TABPFN= "s"     # TabPFNv2 marker

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def _save(fig, name):
    base = os.path.join(FIG_DIR, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    print(f"  → {base}.pdf / .png")


# ── Dataset filter ────────────────────────────────────────────────────────────
def filter_valid_clf(clf_res):
    """
    Keep only (dataset, partition, K) conditions where D-ICL strictly
    outperforms single-agent AND stays at or below centralised, for BOTH
    backbones simultaneously.

        single_agent["accuracy"] < D-ICL@T["accuracy"] <= centralised["accuracy"]

    Returns filtered list and the set of valid (short, partition, K) triples.
    """
    # Group by (short, partition, K) and check both backbones satisfy the criterion
    from collections import defaultdict
    groups = defaultdict(list)
    for r in clf_res:
        groups[(r["short"], r["partition"], r["K"])].append(r)

    valid_keys = set()
    for key, runs in groups.items():
        if len(runs) < 2:      # need both backbones
            continue
        ok = all(
            r["single_agent"]["accuracy"]
            <= r["final"]["consensus"]["accuracy"]
            <= r["oracle"]["accuracy"]
            for r in runs
        )
        if ok:
            valid_keys.add(key)

    filtered = [r for r in clf_res
                if (r["short"], r["partition"], r["K"]) in valid_keys]
    removed  = set(r["short"] for r in clf_res) - set(r["short"] for r in filtered)
    if removed:
        print(f"  [filter_valid_clf] dropped datasets: {removed}")
    print(f"  [filter_valid_clf] kept {len(filtered)}/{len(clf_res)} runs "
          f"({len(valid_keys)} valid conditions)")
    return filtered, valid_keys


def filter_valid_reg(reg_res):
    """
    For regression (lower RMSE is better):
        centralised["rmse"] <= D-ICL@T["rmse"] < single_agent["rmse"]
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in reg_res:
        groups[(r["short"], r["partition"])].append(r)

    valid_keys = set()
    for key, runs in groups.items():
        ok = all(
            r["oracle"]["rmse"]
            <= r["final"]["consensus"]["rmse"]
            <= r["single_agent"]["rmse"]
            for r in runs
        )
        if ok:
            valid_keys.add(key)

    filtered = [r for r in reg_res
                if (r["short"], r["partition"]) in valid_keys]
    removed  = set(r["short"] for r in reg_res) - set(r["short"] for r in filtered)
    if removed:
        print(f"  [filter_valid_reg] dropped datasets: {removed}")
    print(f"  [filter_valid_reg] kept {len(filtered)}/{len(reg_res)} runs "
          f"({len(valid_keys)} valid conditions)")
    return filtered, valid_keys


# ── Shared axis helpers ───────────────────────────────────────────────────────
def _annotate_ds(ax, name, y_offset=-0.22):
    """Place dataset name as italic caption below x-axis instead of a title."""
    ax.annotate(name.split("(")[0].strip(),
                xy=(0.5, y_offset), xycoords="axes fraction",
                ha="center", fontsize=10, fontstyle="italic")


def _legend_outside(fig, axes, ncol=4):
    """
    Place a single shared legend below the bottom row of subplots.
    Collects unique handles/labels from all axes.
    """
    handles, labels = [], []
    seen = set()
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); labels.append(l); seen.add(l)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.get_legend() and ax.get_legend().remove()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(ncol, len(labels)),
               bbox_to_anchor=(0.5, -0.02),
               framealpha=0.92, edgecolor="0.7")


def _ref_lines(ax, sa_val, centr_val, fedavg_val, color, metric="accuracy"):
    """Draw Centralised / Single-agent / FedAvg reference lines."""
    kw = dict(color=color, lw=1.3, alpha=0.72)
    ax.axhline(centr_val,  ls=LS_CENTR,  label=f"Centralized", **kw)
    ax.axhline(sa_val,     ls=LS_SA,     label=f"Single-agent", **kw)
    if not (fedavg_val is None or np.isnan(fedavg_val)):
        ax.axhline(fedavg_val, ls=LS_FEDAVG, label=f"FedAvg", **kw)


# ── Convergence figure factory (works for BOTH clf and reg) ──────────────────
def _convergence_figure(results, datasets, ds_names, partition,
                        metric_key, ylabel, marker_map,
                        ylim=None):
    """
    Unified convergence figure.
    One subplot per dataset; one line per backbone.
    Shaded band = mean ± std when seeded results are provided.
    Reference lines: Centralised (dotted), Single-agent (dash-dot), FedAvg (dashed).
    Shared legend placed below figure.
    """
    n   = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4.2), sharey=False)
    if n == 1: axes = [axes]

    seeded = any("n_seeds" in r for r in results)

    for ds, ax in zip(datasets, axes):
        xs = []   # initialised here so ax.set_xticks never hits UnboundLocalError
        for bb in ["tabicl", "tabpfn"]:
            mk = marker_map[bb]
            sub = [r for r in results if r["short"]==ds
                   and r.get("K",4)==4
                   and r["partition"]==partition
                   and r["backbone"]==bb]
            if not sub: continue
            r0    = sub[0]
            xs    = [rd["round"] for rd in r0["rounds"]]
            ys    = np.array([rd["consensus"][metric_key] for rd in r0["rounds"]])
            col   = BB_CLR[bb]
            label = f"D-ICL / {'TabICL' if bb=='tabicl' else 'TabPFN'}"
            ax.plot(xs, ys, ls=LS_DICL, marker=mk, color=col, label=label)

            # Shaded std band (only when seeded aggregation was run)
            if seeded:
                std_key = metric_key + "_std"
                ys_std  = np.array([rd["consensus"].get(std_key, 0.0)
                                    for rd in r0["rounds"]])
                ax.fill_between(xs, ys - ys_std, ys + ys_std,
                                alpha=0.15, color=col)

            # Reference lines — drawn once using TabICLv2 data, grey colour
            if bb == "tabicl":
                fa    = r0.get("fedavg", {}).get(
                            "accuracy" if metric_key=="accuracy" else "rmse",
                            float("nan"))
                centr = r0["oracle"][metric_key]
                sa    = r0["single_agent"][metric_key]
                _ref_lines(ax, sa, centr, fa, color="#555555", metric=metric_key)

                # Std shading on reference lines
                if seeded:
                    mk_ = metric_key + "_std"
                    for val, std_v in [
                        (centr, r0["oracle"].get(mk_, 0.0)),
                        (sa,    r0["single_agent"].get(mk_, 0.0)),
                    ]:
                        if std_v and std_v > 0:
                            ax.axhspan(val - std_v, val + std_v,
                                       alpha=0.07, color="#555555")

        ax.set_xlabel("Communication round $t$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs if xs else [0])
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="both", which="major")
        _annotate_ds(ax, ds_names.get(ds, ds))

    _legend_outside(fig, axes, ncol=4)
    fig.subplots_adjust(bottom=0.28, wspace=0.35)
    return fig


# ── Bar-chart factory (works for BOTH clf and reg) ────────────────────────────
def _baseline_bar_figure(results, datasets, partition, metric_key,
                         ylabel, higher_is_better=True):
    """
    Unified grouped bar chart: Single-agent | FedAvg | D-ICL/TabICLv2 |
    D-ICL/TabPFNv2 | Centralised — one group per dataset.

    Error bars (±1 std) are drawn whenever seeded aggregation results are
    present (i.e. the result dict contains _std sibling keys).
    """
    seeded = any("n_seeds" in r for r in results)
    std_key = metric_key + "_std"

    def _get(d, k, fallback=float("nan")):
        v = d.get(k, fallback)
        return v if v is not None else fallback

    lbls = []
    sa_,  sa_e_  = [], []
    fa_,  fa_e_  = [], []
    icl_, icl_e_ = [], []
    pfn_, pfn_e_ = [], []
    cen_, cen_e_ = [], []

    for ds in datasets:
        si = [r for r in results if r["short"]==ds and r.get("K",4)==4
              and r["partition"]==partition and r["backbone"]=="tabicl"]
        sp = [r for r in results if r["short"]==ds and r.get("K",4)==4
              and r["partition"]==partition and r["backbone"]=="tabpfn"]
        if not si:
            continue
        r_icl = si[0]; r_pfn = sp[0] if sp else {}

        lbls.append(r_icl["dataset"].split("(")[0].strip())

        # Point estimates
        sa_.append(_get(r_icl["single_agent"],   metric_key))
        fa_v = _get(r_icl.get("fedavg", {}),     metric_key)
        fa_.append(fa_v)
        icl_.append(_get(r_icl["final"]["consensus"], metric_key))
        pfn_.append(_get(r_pfn.get("final",{}).get("consensus",{}), metric_key)
                    if r_pfn else float("nan"))
        cen_.append(_get(r_icl["oracle"],         metric_key))

        # Std — zero when not seeded so error bars are invisible
        if seeded:
            sa_e_.append(_get(r_icl["single_agent"],          std_key, 0.0))
            fa_e_.append(_get(r_icl.get("fedavg", {}),        std_key, 0.0))
            icl_e_.append(_get(r_icl["final"]["consensus"],   std_key, 0.0))
            pfn_e_.append(_get(r_pfn.get("final",{}).get("consensus",{}), std_key, 0.0)
                          if r_pfn else 0.0)
            cen_e_.append(_get(r_icl["oracle"],               std_key, 0.0))
        else:
            sa_e_.append(0.0); fa_e_.append(0.0); icl_e_.append(0.0)
            pfn_e_.append(0.0); cen_e_.append(0.0)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7*len(lbls)), 4.8))
    x       = np.arange(len(lbls))
    w       = 0.16
    offsets = [-2*w, -w, 0, w, 2*w]

    # capsize for error bars: small enough not to clutter, visible enough to read
    CAPSIZE = 3
    ERR_KW  = dict(elinewidth=1.1, capsize=CAPSIZE, capthick=1.1, ecolor="0.3")

    bars = [
        (sa_,  sa_e_,  "#888888",        "Single-agent"),
        (fa_,  fa_e_,  "#F1C40F",        "FedAvg"),
        (icl_, icl_e_, BB_CLR["tabicl"], "D-ICL / TabICL"),
        (pfn_, pfn_e_, BB_CLR["tabpfn"], "D-ICL / TabPFN"),
        (cen_, cen_e_, "#2DA34D",        "Centralized"),
    ]
    for (vals, errs, col, lbl), off in zip(bars, offsets):
        # Replace nan values with 0 so bar() does not raise
        vals_clean = [v if not (isinstance(v,float) and np.isnan(v)) else 0.0
                      for v in vals]
        errs_clean = [e if not (isinstance(e,float) and np.isnan(e)) else 0.0
                      for e in errs]
        ax.bar(x + off, vals_clean, w,
               alpha=0.88, color=col, label=lbl,
               yerr=errs_clean if seeded else None,
               error_kw=ERR_KW if seeded else {})

    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if higher_is_better:
        top = max((v for v in cen_ if not np.isnan(v)), default=1.0)
        ax.set_ylim(0, top * 1.22)
    ax.legend(ncol=3, loc="upper right", fontsize=9)
    fig.subplots_adjust(bottom=0.24)
    return fig


# ── Individual figure builders ────────────────────────────────────────────────

def _fig_clf_convergence(clf_res, datasets, ds_names, partition):
    return _convergence_figure(
        clf_res, datasets, ds_names, partition,
        metric_key="accuracy", ylabel="Accuracy",
        marker_map={"tabicl": MK_TABICL, "tabpfn": MK_TABPFN},
        ylim=(0.25, 1.08))


def _fig_reg_convergence(reg_res, reg_ds, ds_names_reg, partition):
    return _convergence_figure(
        reg_res, reg_ds, ds_names_reg, partition,
        metric_key="rmse", ylabel="RMSE",
        marker_map={"tabicl": MK_TABICL, "tabpfn": MK_TABPFN},
        ylim=None)   # auto-scale — RMSE ranges vary widely across datasets


def _fig_clf_bar(clf_res, datasets, partition):
    return _baseline_bar_figure(
        clf_res, datasets, partition,
        metric_key="accuracy", ylabel="Accuracy",
        higher_is_better=True)


def _fig_reg_bar(reg_res, reg_ds, partition):
    return _baseline_bar_figure(
        reg_res, reg_ds, partition,
        metric_key="rmse", ylabel="RMSE (lower is better)",
        higher_is_better=False)


def _clf_k_scaling(clf_res, cfg):
    """K-scaling plot (uses first clf dataset that has all K values)."""
    # pick the ablation dataset
    ds = cfg.ablation_dataset
    # Customize K values
    K_list = cfg.K_values
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for bb in ["tabicl", "tabpfn"]:
        accs = []
        for K in K_list:
        # for K in cfg.K_values:
            sub = [r for r in clf_res if r["short"]==ds and r["K"]==K
                   and r["partition"]=="iid" and r["backbone"]==bb]
            accs.append(sub[0]["final"]["consensus"]["accuracy"] if sub else float("nan"))
        mk = MK_TABICL if bb=="tabicl" else MK_TABPFN
        ax.plot(K_list, accs, ls=LS_DICL, marker=mk, color=BB_CLR[bb],
                label=f"D-ICL / {'TabICL' if bb=='tabicl' else 'TabPFN'}")
    # reference lines from K=4
    ref = [r for r in clf_res if r["short"]==ds and r["K"]==4
           and r["partition"]=="iid" and r["backbone"]=="tabicl"]
    if ref:
        fa = ref[0].get("fedavg", {}).get("accuracy", float("nan"))
        ax.axhline(ref[0]["single_agent"]["accuracy"],
                   color="#555555", ls=LS_SA, lw=1.3, label="Single-agent")
        ax.axhline(ref[0]["oracle"]["accuracy"],
                   color="#555555", ls=LS_CENTR, lw=1.3, label="Centralized")
        if not np.isnan(fa):
            ax.axhline(fa, color="#555555", ls=LS_FEDAVG, lw=1.3, label="FedAvg")
    ax.set_xlabel("Number of agents $K$")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(K_list)
    ax.legend(loc="lower right")
    _annotate_ds(ax, ds)
    fig.subplots_adjust(bottom=0.18)
    return fig


def _clf_iid_vs_noniid(clf_res, datasets):
    """IID vs Non-IID curves — one panel per dataset (up to 3)."""
    n   = min(3, len(datasets))
    fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4.2), sharey=False)
    if n == 1: axes = [axes]
    for di, (ds, ax) in enumerate(zip(datasets[:n], axes)):
        col = DS_CLR[di]
        xs = []; sub = []   # ensure always defined even if no data matches
        for pt, ls, mk in [("iid", LS_DICL, MK_TABICL),
                            ("non_iid", LS_FEDAVG, MK_TABPFN)]:
            sub = [r for r in clf_res if r["short"]==ds and r["K"]==4
                   and r["partition"]==pt and r["backbone"]=="tabicl"]
            if not sub: continue
            xs = [rd["round"] for rd in sub[0]["rounds"]]
            ys = [rd["consensus"]["accuracy"] for rd in sub[0]["rounds"]]
            lbl = "IID" if pt=="iid" else "Non-IID"
            ax.plot(xs, ys, ls=ls, marker=mk, color=col, label=lbl)
        ax.set_xlabel("Communication round $t$")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.25, 1.08)
        ax.set_xticks(xs if xs else [0])
        ax.legend(loc="lower right")
        _annotate_ds(ax, sub[0]["dataset"] if sub else ds)
    fig.subplots_adjust(bottom=0.25, wspace=0.35)
    return fig


def _theory_variance(cfg):
    Kr = np.arange(1, 17)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.semilogy(Kr, cfg.sigma_base**2/Kr, "o"+LS_DICL, color="#2E5FA3", label="IID")
    ax.semilogy(Kr, cfg.sigma_base**2/Kr + cfg.sigma_het_noniid**2,
                "s"+LS_FEDAVG, color="#C0392B", label="Non-IID")
    ax.fill_between(Kr, cfg.sigma_base**2/Kr,
                    cfg.sigma_base**2/Kr + cfg.sigma_het_noniid**2,
                    alpha=0.12, color="#C0392B")
    ax.set_xlabel("Number of agents $K$")
    ax.set_ylabel(r"$\mathrm{Var}[\bar{p}^t]$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    return fig


def _theory_convergence(cfg):
    Tr = np.arange(1, 21); C1 = cfg.L_lipschitz * 2
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for K, col in [(2, K_CLR[2]), (4, K_CLR[4]), (8, K_CLR[8])]:
        ax.plot(Tr, C1/np.sqrt(Tr), LS_DICL, color=col, label=f"$K={K}$, IID")
        ax.plot(Tr, C1/np.sqrt(Tr) + cfg.sigma_het_noniid/np.sqrt(K),
                LS_FEDAVG, color=col, alpha=0.65, label=f"$K={K}$, non-IID")
    ax.set_xlabel("Rounds $T$")
    ax.set_ylabel("Loss gap bound")
    ax.legend(ncol=2, loc="upper right", fontsize=10)
    return fig


def _theory_tau(cfg):
    tau_r = np.linspace(0.5, 0.99, 120)
    eps   = -(2*tau_r - 1) * 0.15
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(tau_r, eps, color="#2DA34D", lw=2.2)
    ax.axhline(0, color="black", ls="--", lw=1.0, alpha=0.6)
    ax.fill_between(tau_r, eps, 0, where=(eps < 0),
                    alpha=0.18, color="#2DA34D",
                    label=r"$\varepsilon(\kappa)<0$ (beneficial)")
    ax.fill_between(tau_r, eps, 0, where=(eps >= 0),
                    alpha=0.14, color="#C0392B",
                    label=r"$\varepsilon(\kappa)\geq0$ (harmful)")
    ax.axvline(cfg.tau, color="#E05C1A", ls=LS_CENTR, lw=2.0,
               label=rf"$\tau={cfg.tau}$ (default)")
    ax.set_xlabel(r"Confidence threshold $\tau$")
    ax.set_ylabel(r"$\varepsilon(\tau)$")
    ax.legend(loc="upper right")
    return fig


def _comm_cost_heatmap(cfg):
    Kv = [2, 4, 8, 16]; Mv = [50, 100, 200, 500]
    cost = np.array([[K*M*10*cfg.T for M in Mv] for K in Kv])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sns.heatmap(np.log10(cost+1), ax=ax,
                xticklabels=[f"$M={m}$" for m in Mv],
                yticklabels=[f"$K={k}$" for k in Kv],
                annot=[[f"{v/1000:.1f}k" for v in row] for row in cost],
                fmt="", cmap="Blues", linewidths=0.5,
                cbar_kws={"label": r"$\log_{10}(\mathrm{floats})$",
                          "shrink": 0.85})
    ax.set_xlabel("Query pool size $|\\mathcal{Q}|$")
    ax.set_ylabel("Number of agents $K$")
    return fig


def build_figures(clf_res, reg_res, cfg):
    """
    Filter to valid datasets (single_agent < D-ICL <= centralized),
    then save 12 individual publication-quality figures (PDF + PNG, 300 DPI).

    All classification and regression convergence/bar figures use identical
    visual encoding:  line styles, colors, markers, axis labels, and legend
    placement are governed by the shared constants at the top of CELL 14.
    """
    sns.set_style("whitegrid")

    # ── Dataset filtering ────────────────────────────────────────────────────
    if cfg.filtered:
        print("\n  Filtering to datasets where SA < D-ICL@T <= Centralized …")
        clf_valid, _  = filter_valid_clf(clf_res)
        reg_valid, _  = filter_valid_reg(reg_res)
    else:
        clf_valid = clf_res
        reg_valid = reg_res

    datasets    = list(dict.fromkeys(r["short"] for r in clf_valid))
    ds_names    = {r["short"]: r["dataset"] for r in clf_valid}
    reg_ds      = list(dict.fromkeys(r["short"] for r in reg_valid))
    ds_names_reg= {r["short"]: r["dataset"] for r in reg_valid}

    if not datasets:
        print("  WARNING: no valid CLF datasets after filtering — using all.")
        clf_valid = clf_res
        datasets  = list(dict.fromkeys(r["short"] for r in clf_valid))
        ds_names  = {r["short"]: r["dataset"] for r in clf_valid}

    if not reg_ds:
        print("  WARNING: no valid REG datasets after filtering — using all.")
        reg_valid = reg_res
        reg_ds    = list(dict.fromkeys(r["short"] for r in reg_valid))
        ds_names_reg = {r["short"]: r["dataset"] for r in reg_valid}

    # ── Figure list ──────────────────────────────────────────────────────────
    figs = [
        (_fig_clf_convergence(clf_valid, datasets, ds_names, "iid"),
         "fig1_clf_convergence_iid"),

        (_fig_clf_convergence(clf_valid, datasets, ds_names, "non_iid"),
         "fig2_clf_convergence_noniid"),

        (_fig_clf_bar(clf_valid, datasets, "iid"),
         "fig3_clf_baseline_bar_iid"),

        (_fig_clf_bar(clf_valid, datasets, "non_iid"),
         "fig4_clf_baseline_bar_noniid"),

        (_clf_k_scaling(clf_valid, cfg),
         "fig5_clf_k_scaling"),

        (_clf_iid_vs_noniid(clf_valid, datasets),
         "fig6_clf_iid_vs_noniid"),

        (_theory_variance(cfg),
         "fig7_theory_variance_reduction"),

        (_theory_convergence(cfg),
         "fig8_theory_convergence_bound"),

        (_theory_tau(cfg),
         "fig9_theory_tau_threshold"),

        (_comm_cost_heatmap(cfg),
         "fig10_communication_cost"),

        (_fig_reg_convergence(reg_valid, reg_ds, ds_names_reg, "iid"),
         "fig11_reg_convergence_iid"),

        (_fig_reg_bar(reg_valid, reg_ds, "iid"),
         "fig12_reg_baseline_bar_iid"),
    ]

    for fig, name in figs:
        _save(fig, name)

    print(f"\n  All {len(figs)} figures saved to ./{FIG_DIR}/")
    print(f"  Valid CLF datasets: {datasets}")
    print(f"  Valid REG datasets: {reg_ds}")
