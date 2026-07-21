#!/usr/bin/env python3
"""Generate the mechanism figures for the LRTC paper (docs/paper/paper-02/).

Reads the cross-population result JSONs and renders three publication figures:
  F_dissociation : per-FM best-probe R^2, DFA vs 1/f, on CAUEEG and BrainLat.
                   Shows the raw-vs-spectral split on 1/f and universal DFA failure.
  F_prepool      : per-FM ordered vs shuffled pre-pool CNN R^2 (5-seed mean+/-SD),
                   the order gap ~= 0 for every model.
  F_transfer     : DFA cross-population transfer AUROC per direction with resample
                   CI and the permutation null band; FM cells at chance.

Run from repo root:  python scripts/analysis/plot_lrtc_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XP = ROOT / "results" / "cross_population"
OUT = ROOT / "docs" / "paper" / "paper-02" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RAW = {"REVE", "LaBraM", "BENDR"}  # raw-waveform front end
COL_RAW, COL_SPEC, COL_CLASS = "#c0392b", "#2962ff", "#27ae60"


def load(name: str) -> dict:
    return json.loads((XP / name).read_text())


def best_r2(entry: dict, target: str) -> float:
    """Best (max) CV R^2 across probes for a target; DFA = max(DFA_full, DFA_short)."""
    if target == "DFA":
        vals = []
        for t in ("DFA_full", "DFA_short"):
            if t in entry:
                vals.append(max(entry[t].values()))
        return max(vals) if vals else float("nan")
    if target in entry:
        return max(entry[target].values())
    return float("nan")


def fm_name(k: str) -> str:
    return k.replace("_pooled", "")


# ------------------------------------------------------------------ F_dissociation
def fig_dissociation() -> None:
    brain = load("brainlat_multifm_encoding_probe.json")["cv_r2"]
    caueeg = load("multifm_lrtc_encoding_fullN.json")["cv_r2"]  # N=770
    bendr_caueeg = load("extra_fm_lrtc_probe.json")["cv_r2"]["BENDR"]  # N=200, not rerun
    caueeg = {**caueeg, "BENDR": bendr_caueeg}
    order = ["REVE", "LaBraM", "BENDR", "CBraMod", "BIOT", "classical"]

    def series(d: dict, target: str) -> list[float]:
        out = []
        for m in order:
            key = "REVE_pooled" if (m == "REVE" and "REVE_pooled" in d) else m
            out.append(best_r2(d.get(key, {}), target) if key in d else np.nan)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (dd, title) in zip(
        axes, [(caueeg, "CAUEEG (Korean, N=770$^{\\dagger}$)"), (brain, "BrainLat (Latin-American, N=79)")]
    ):
        dfa = series(dd, "DFA")
        aper = series(dd, "1/f")
        x = np.arange(len(order))
        w = 0.38
        ax.bar(x - w / 2, dfa, w, label="DFA (temporal)", color="#8e8e8e", edgecolor="k", linewidth=0.5)
        ax.bar(x + w / 2, aper, w, label="1/f (spectral)", color="#f2b134", edgecolor="k", linewidth=0.5)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("best CV $R^2$")
        for i, m in enumerate(order):
            c = COL_CLASS if m == "classical" else (COL_RAW if m in RAW else COL_SPEC)
            ax.get_xticklabels()[i].set_color(c)
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle(
        "Spectral-vs-temporal dissociation: 1/f split by front end, DFA recovered by no FM\n"
        "$\\dagger$BENDR's CAUEEG bars use the original N=200 subsample (not rerun)",
        fontsize=10,
    )
    fig.tight_layout()
    p = OUT / "fig_dissociation.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


# ------------------------------------------------------------------ F_prepool
def fig_prepool() -> None:
    pp = load("multifm_prepool_probe.json")["prepool"]
    # REVE reported from its finer token-level probe (single-seed ~ -0.02 / -0.02).
    reve = {"cnn_ordered_mean": -0.02, "cnn_ordered_std": 0.0,
            "cnn_shuffled_mean": -0.02, "cnn_shuffled_std": 0.0}
    models = ["REVE", "LaBraM", "BENDR", "CBraMod", "BIOT"]
    data = {"REVE": reve, **{m: pp[m] for m in models if m in pp}}

    x = np.arange(len(models))
    w = 0.38
    ordered = [data[m]["cnn_ordered_mean"] for m in models]
    ordered_e = [data[m].get("cnn_ordered_std", 0) for m in models]
    shuf = [data[m]["cnn_shuffled_mean"] for m in models]
    shuf_e = [data[m].get("cnn_shuffled_std", 0) for m in models]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w / 2, ordered, w, yerr=ordered_e, capsize=3, label="ordered", color="#34495e", edgecolor="k", linewidth=0.5)
    ax.bar(x + w / 2, shuf, w, yerr=shuf_e, capsize=3, label="shuffled", color="#bdc3c7", edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(0.32, color=COL_CLASS, ls="--", lw=1.2, label="classical (DFA)")
    ax.set_xticks(x)
    for i, m in enumerate(models):
        ax.get_xticklabels()
    ax.set_xticklabels(models, fontsize=9)
    for i, m in enumerate(models):
        ax.get_xticklabels()[i].set_color(COL_RAW if m in RAW else COL_SPEC)
    ax.set_ylabel("pre-pool CNN CV $R^2$ (DFA)")
    ax.set_title(
        "Order control: REVE/CBraMod order-independent; BIOT gap small but\n"
        "nominally reliable ($p$=0.049); LaBraM/BENDR probe underperforms (not resolved)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = OUT / "fig_prepool_order.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


# ------------------------------------------------------------------ F_transfer
def fig_transfer() -> None:
    pw = load("transfer_3cohort.json")["pairwise"]
    dirs = ["K->W", "W->K", "B->W", "B->K"]
    keys = [f"DFA|{d}" for d in dirs]
    auroc = [pw[k]["point_auroc"] for k in keys]
    ci = [pw[k]["resample_ci95"] for k in keys]
    pvals = [pw[k]["p_above_chance"] for k in keys]
    pholm = [pw[k]["p_holm"] for k in keys]

    x = np.arange(len(dirs))
    yerr = np.array([[a - c[0] for a, c in zip(auroc, ci)],
                     [c[1] - a for a, c in zip(auroc, ci)]])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.errorbar(x, auroc, yerr=yerr, fmt="o", color="#8e44ad", capsize=4,
                markersize=8, label="DFA feature", lw=1.5)
    ax.axhspan(0.0, 0.44, color=COL_RAW, alpha=0.12)
    ax.axhline(0.44, color=COL_RAW, ls="--", lw=1.2, label="frozen FM ceiling ($\\leq$0.44)")
    ax.axhline(0.5, color="k", ls=":", lw=1.0, label="chance")
    for xi, (a, p, ph) in enumerate(zip(auroc, pvals, pholm)):
        star = "*" if ph < 0.05 else ""
        ax.annotate(f"{a:.3f}{star}\n$p$={p:.3f}, $p_{{holm}}$={ph:.3f}", (xi, a),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dirs, fontsize=10)
    ax.set_ylim(0.35, 0.92)
    ax.set_ylabel("cross-population AUROC")
    ax.set_xlabel("transfer direction (W=Greek, K=Korean, B=BrainLat)")
    ax.set_title(
        "DFA transfers where the frozen FM is at chance\n"
        "(directional; no cell clears Holm-adjusted $p_{holm}<0.05$, * marks it if one does)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = OUT / "fig_transfer.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def main() -> None:
    print("Generating LRTC paper figures ->", OUT.relative_to(ROOT))
    fig_dissociation()
    fig_prepool()
    fig_transfer()
    print("done.")


if __name__ == "__main__":
    main()
