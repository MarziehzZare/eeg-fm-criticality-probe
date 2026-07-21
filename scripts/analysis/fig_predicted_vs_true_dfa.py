"""Predicted-vs-true scatter for the classical DFA_full positive control on CAUEEG.

Addresses reviewer point #6: the DFA target has narrow variance (exponent in
[0.5,1.0] for 96-99% of subjects), so R^2 alone is hard for a reader to judge --
a scatter of cross-validated predictions against ground truth lets the reader see
whether the classical R^2=0.28 reflects real structure or a narrow-band fit, and
gives a visual contrast against the FMs' R^2~=0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

import scripts.analysis.fm_lrtc_nonlinear_probe as p01  # noqa: E402
from src.io_safety import safe_np_load  # noqa: E402

CACHE = ROOT / "results/cross_population/p01_targets_N1187.npz"
OUT = ROOT / "docs/paper/paper-02/figures/fig_predicted_vs_true_dfa.png"


def main() -> None:
    tcache = {k: list(v) for k, v in safe_np_load(str(CACHE))["data"].item().items()}
    enc = p01.load_embeddings()
    common = set(enc["REVE"]) & set(enc["classical"])
    serials = [s for s in tcache if s in common]
    y = np.array([tcache[s][0] for s in serials])  # DFA_full
    Xc = np.vstack([enc["classical"][s] for s in serials])
    Xr = np.vstack([enc["REVE"][s] for s in serials])
    n = len(serials)
    print(f"N={n}")

    probes = p01.make_probes(Xc.shape[1], n)
    classical_pipe = probes["ridge_linear_full"]()
    reve_probes = p01.make_probes(Xr.shape[1], n)
    reve_pipe = reve_probes["ridge_linear_full"]()

    cv = KFold(5, shuffle=True, random_state=0)
    pred_classical = cross_val_predict(classical_pipe, Xc, y, cv=cv)
    pred_reve = cross_val_predict(reve_pipe, Xr, y, cv=cv)

    from sklearn.metrics import r2_score
    r2_c = r2_score(y, pred_classical)
    r2_r = r2_score(y, pred_reve)
    print(f"classical R2={r2_c:.3f}  REVE R2={r2_r:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    lims = (y.min() - 0.05, y.max() + 0.05)
    for ax, pred, title, r2, color in [
        (axes[0], pred_classical, "Classical (positive control)", r2_c, "#2c7a3f"),
        (axes[1], pred_reve, "REVE (frozen)", r2_r, "#8e44ad"),
    ]:
        ax.scatter(y, pred, s=10, alpha=0.35, color=color)
        ax.plot(lims, lims, "k--", lw=1, label="identity")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("true DFA$_{full}$")
        ax.set_title(f"{title}\n$R^2={r2:.3f}$, $N={n}$", fontsize=10)
        ax.set_aspect("equal")
    axes[0].set_ylabel("predicted DFA$_{full}$ (5-fold CV)")
    fig.suptitle("Predicted vs. true alpha-envelope DFA exponent, CAUEEG (full-N rerun)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
