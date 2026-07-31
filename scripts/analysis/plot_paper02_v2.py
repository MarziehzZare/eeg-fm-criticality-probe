"""Generate Paper 02 figures directly from revised result JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.io_safety import safe_np_load  # noqa: E402


RESULTS = ROOT / "results/paper02"
FIGURES = ROOT / "docs/paper/paper-02/figures"
MODELS = ["REVE", "LaBraM", "BENDR", "CBraMod", "BIOT", "classical"]
LABELS = {
    "REVE": "REVE",
    "LaBraM": "LaBraM",
    "BENDR": "BENDR",
    "CBraMod": "CBraMod",
    "BIOT": "BIOT",
    "classical": "Classical",
}
COLORS = {
    "REVE": "#4C78A8",
    "LaBraM": "#6B8EAD",
    "BENDR": "#8BA6B8",
    "CBraMod": "#8F63A9",
    "BIOT": "#E07B39",
    "classical": "#3E8E68",
}


def _load(cohort: str) -> dict:
    return json.loads((RESULTS / f"primary_encoding_v2_{cohort}.json").read_text())


def _errorbar_panel(axis, data: dict, title: str, threshold: bool) -> None:
    position = np.arange(len(MODELS))
    point = np.asarray([data[model]["r2"] for model in MODELS])
    low = np.asarray([data[model]["r2_ci95"][0] for model in MODELS])
    high = np.asarray([data[model]["r2_ci95"][1] for model in MODELS])
    axis.errorbar(
        position,
        point,
        yerr=np.vstack([point - low, high - point]),
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=3,
        zorder=2,
    )
    axis.scatter(
        position,
        point,
        s=55,
        c=[COLORS[model] for model in MODELS],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    axis.axhline(0, color="#333333", linewidth=0.8)
    if threshold:
        axis.axhline(
            0.10,
            color="#B23A48",
            linestyle="--",
            linewidth=1,
            label="descriptive $R^2=0.10$ reference",
        )
        axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.set_xticks(position, [LABELS[model] for model in MODELS], rotation=35, ha="right")
    axis.set_ylabel("Out-of-fold $R^2$ (conditional 95% interval)")
    axis.set_title(title)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)


def plot_encoding() -> Path:
    brainlat = _load("brainlat")
    caueeg = _load("caueeg")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.3), constrained_layout=True)
    _errorbar_panel(axes[0, 0], caueeg["models"], "CAUEEG: DFA v2", True)
    _errorbar_panel(axes[0, 1], brainlat["models"], "BrainLat: DFA v2", True)
    _errorbar_panel(
        axes[1, 0],
        caueeg["aperiodic_comparison"]["models"],
        "CAUEEG: aperiodic exponent",
        False,
    )
    _errorbar_panel(
        axes[1, 1],
        brainlat["aperiodic_comparison"]["models"],
        "BrainLat: aperiodic exponent",
        False,
    )
    for label, axis in zip(("A", "B", "C", "D"), axes.ravel()):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
        )
    path = FIGURES / "fig_dissociation_v2.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_predictions() -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(8, 7.2), constrained_layout=True)
    for row, cohort in enumerate(("caueeg", "brainlat")):
        result = _load(cohort)
        for column, model in enumerate(("classical", "REVE")):
            data = result["oof_predictions"][model]
            target = np.asarray(data["target"])
            prediction = np.asarray(data["prediction"])
            axis = axes[row, column]
            axis.scatter(
                target,
                prediction,
                s=15,
                alpha=0.55,
                color=COLORS[model],
                edgecolor="none",
            )
            lower = min(target.min(), prediction.min())
            upper = max(target.max(), prediction.max())
            axis.plot([lower, upper], [lower, upper], color="#333333", linestyle="--")
            r2 = result["models"][model]["r2"]
            axis.set_title(f"{cohort.upper()} — {LABELS[model]} ($R^2={r2:+.2f}$)")
            axis.set_xlabel("Observed DFA v2")
            axis.set_ylabel("Out-of-fold prediction")
            axis.grid(color="#E1E1E1", linewidth=0.5)
    path = FIGURES / "fig_predicted_vs_true_dfa_v2.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_target_qc() -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 6.4), constrained_layout=True)
    metrics = (
        ("exponent", "Channel DFA exponent"),
        ("fit_r2", "Log--log fit $R^2$"),
        ("retained_fraction", "Retained-sample fraction"),
        ("alpha_peak_snr", "Alpha peak / aperiodic background"),
    )
    for cohort, color in (("caueeg", "#4C78A8"), ("brainlat", "#E07B39")):
        data = safe_np_load(str(RESULTS / f"dfa_v2_{cohort}.npz"))
        eligible = np.asarray(data["eligible"], dtype=bool)
        for axis, (key, label) in zip(axes.ravel(), metrics):
            values = np.asarray(data[key], dtype=float)[eligible].ravel()
            values = values[np.isfinite(values)]
            axis.hist(
                values,
                bins=35,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=color,
                label=cohort.upper(),
            )
            axis.set_xlabel(label)
            axis.set_ylabel("Density")
            axis.grid(axis="y", color="#E1E1E1", linewidth=0.5)
    axes[0, 0].legend(frameon=False)
    path = FIGURES / "fig_target_qc_v2.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_prepool_order() -> Path | None:
    result_path = RESULTS / "prepool_order_permutation_5model.json"
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text())
    result = payload["primary_bs128"]
    sensitivity = payload.get("sensitivity_bs32", {})
    stability = payload.get("configuration_sensitivity_12cfg_K30", {})
    stability_bs128 = stability.get("batch_128", {})
    stability_bs32 = stability.get("batch_32", {})
    models = ["REVE", "LaBraM", "BENDR", "CBraMod", "BIOT"]
    position = np.arange(len(models))
    aggregate = np.asarray(
        [result[model]["observed_minus_null_mean"] for model in models]
    )

    figure, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for model_idx, model in enumerate(models):
        if model in stability_bs128:
            config_gaps = np.asarray(
                stability_bs128[model]["config_order_increments"],
                dtype=float,
            )
        else:
            row = result[model]
            ordered = np.asarray(row["ordered_r2_per_config"], dtype=float)
            null_matrix = np.asarray(row["null_matrix_per_config"], dtype=float)
            config_gaps = ordered - null_matrix.mean(axis=0)
        jitter = np.linspace(-0.10, 0.10, len(config_gaps))
        axis.scatter(
            np.full(len(config_gaps), model_idx) + jitter,
            config_gaps,
            s=32,
            color="#A7A7A7",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="configuration-specific gap" if model_idx == 0 else None,
        )

    axis.scatter(
        position,
        aggregate,
        marker="D",
        s=60,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
        label="4-configuration mean (batch 128; K=150)",
        zorder=5,
    )
    for model_idx, model in enumerate(models):
        if model not in stability_bs128:
            continue
        axis.scatter(
            model_idx,
            stability_bs128[model]["observed_minus_null_mean"],
            marker="^",
            s=65,
            facecolor="#2A9D8F",
            edgecolor="white",
            linewidth=0.7,
            label=(
                "12-configuration mean (batch 128; K=30)"
                if model == "CBraMod"
                else None
            ),
            zorder=5,
        )
    for model_idx, model in enumerate(models):
        if model in stability_bs32:
            batch32_mean = stability_bs32[model]["observed_minus_null_mean"]
            batch32_label = "12-configuration mean (batch 32; K=30)"
        elif model in sensitivity:
            batch32_mean = sensitivity[model]["observed_minus_null_mean"]
            batch32_label = "4-configuration mean (batch 32)"
        else:
            continue
        axis.scatter(
            model_idx,
            batch32_mean,
            marker="s",
            s=55,
            facecolor="none",
            edgecolor="#E07B39",
            linewidth=1.5,
            label=batch32_label if model == "CBraMod" else None,
            zorder=5,
        )
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(position, models)
    axis.set_ylabel("Ordered minus shuffled-reference mean $\\Delta R^2$")
    axis.grid(axis="y", color="#E1E1E1", linewidth=0.5)
    axis.legend(frameon=False, ncol=1, fontsize=8)
    path = FIGURES / "fig_prepool_order_v2.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [plot_encoding(), plot_predictions(), plot_target_qc(), plot_prepool_order()]
    for path in paths:
        if path is not None:
            print(path)


if __name__ == "__main__":
    main()
