"""Fixed primary encoding analysis for Paper 02.

Unlike the legacy figure, this analysis does not maximize across target scale
ranges or probe families.  It uses one target (the v2 alpha-envelope DFA
estimand) and one readout (scaler -> PCA-50 -> ridge), with hyperparameter
selection confined to each outer training fold.

The output includes subject-level out-of-fold predictions and a stratified
pair-bootstrap interval conditional on the fixed prediction vector. It does
not refit the overlapping CV training folds and is not a full-pipeline
generalization interval. R2=0.10 is retained only as a descriptive
reference adopted during revision; it is not a prespecified confirmatory
decision boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.io_safety import safe_np_load  # noqa: E402


TARGET_ROOT = ROOT / "results/paper02"
RESULT_ROOT = ROOT / "results/paper02"
DESCRIPTIVE_REFERENCE_R2 = 0.10
OUTER_FOLDS = 5
SEED = 20260729


def _flat(value) -> np.ndarray:
    value = value.item() if getattr(value, "shape", None) == () else value
    if isinstance(value, dict):
        for key in ("embedding", "mean_embed"):
            if key in value:
                value = value[key]
                break
    return np.asarray(value, dtype=np.float64).ravel()


def _load_caueeg_embeddings() -> dict[str, dict[str, np.ndarray]]:
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    reve = safe_np_load(
        str(ROOT / "data/embeddings/caueeg_3way_reve_19ch_subjects.npz")
    )["data"].item()
    embeddings["REVE"] = {
        serial: _flat(value) for serial, value in reve.items()
    }

    labram = safe_np_load(
        str(ROOT / "data/embeddings/caueeg_3way_labram_multiepoch_subjects.npz")
    )["data"].item()
    embeddings["LaBraM"] = {
        serial: _flat(value) for serial, value in labram.items()
    }

    multi = safe_np_load(
        str(ROOT / "data/embeddings/caueeg_3way_multiepoch_19ch_all.npz")
    )["data"].item()
    for source, name in (
        ("classical", "classical"),
        ("cbramod", "CBraMod"),
        ("biot", "BIOT"),
    ):
        embeddings[name] = {
            serial: _flat(value)
            for serial, value in multi[source]["subject_mean_embeds"].items()
        }

    extra = safe_np_load(
        str(ROOT / "results/cross_population/extra_fm_embeds.npz")
    )["data"].item()
    embeddings["BENDR"] = {
        str(serial): _flat(value["BENDR"])
        for serial, value in extra.items()
        if "BENDR" in value
    }
    return embeddings


def _load_brainlat_embeddings() -> dict[str, dict[str, np.ndarray]]:
    reve_cache = safe_np_load(
        str(ROOT / "results/cross_population/brainlat_reve_encoding.npz")
    )
    serials = [str(value) for value in reve_cache["serials"]]
    embeddings = {
        "REVE": dict(zip(serials, np.asarray(reve_cache["pooled"]))),
        "classical": dict(zip(serials, np.asarray(reve_cache["classical"]))),
    }
    multifm = safe_np_load(
        str(ROOT / "results/cross_population/brainlat_multifm_embeds.npz")
    )["data"].item()
    for model in ("LaBraM", "CBraMod", "BIOT", "BENDR"):
        embeddings[model] = {
            str(serial): _flat(value)
            for serial, value in multifm[model].items()
        }
    return embeddings


LOADERS = {
    "caueeg": _load_caueeg_embeddings,
    "brainlat": _load_brainlat_embeddings,
}


def _load_aperiodic_target(cohort: str) -> dict[str, float]:
    """Load the existing fixed-mode 1--40 Hz specparam comparison target."""

    if cohort == "brainlat":
        data = safe_np_load(
            str(ROOT / "results/cross_population/brainlat_reve_encoding.npz")
        )
        return {
            str(serial): float(value)
            for serial, value in zip(data["serials"], data["f1"])
        }
    cache = safe_np_load(
        str(ROOT / "results/cross_population/p01_targets.npz")
    )["data"].item()
    return {
        str(serial): float(values[2])
        for serial, values in cache.items()
        if len(values) >= 3
    }


def _load_target(cohort: str) -> dict:
    path = TARGET_ROOT / f"dfa_v2_{cohort}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run paper02_extract_dfa_v2.py first"
        )
    data = safe_np_load(str(path))
    exponent = np.asarray(data["exponent"], dtype=float)
    finite_count = np.isfinite(exponent).sum(axis=1)
    target_value = np.divide(
        np.nansum(exponent, axis=1),
        finite_count,
        out=np.full(len(exponent), np.nan),
        where=finite_count > 0,
    )
    return {
        "serial": np.asarray(data["serial"]).astype(str),
        "label": np.asarray(data["label"], dtype=int),
        "eligible": np.asarray(data["eligible"], dtype=bool),
        "target": target_value,
    }


def _nested_oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    prediction = np.full(len(y), np.nan)
    dummy_prediction = np.full(len(y), np.nan)
    chosen_alpha: list[float] = []
    outer = KFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(outer.split(x)):
        n_components = max(
            2,
            min(50, x.shape[1], len(train) - 2),
        )
        scaler = StandardScaler().fit(x[train])
        train_scaled = scaler.transform(x[train])
        test_scaled = scaler.transform(x[test])
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=SEED + fold,
        ).fit(train_scaled)
        train_reduced = pca.transform(train_scaled)
        test_reduced = pca.transform(test_scaled)
        ridge = RidgeCV(
            alphas=np.logspace(-2, 4, 9),
            cv=None,
        )
        ridge.fit(train_reduced, y[train])
        prediction[test] = ridge.predict(test_reduced)
        dummy_prediction[test] = np.mean(y[train])
        chosen_alpha.append(float(ridge.alpha_))
    return prediction, dummy_prediction, chosen_alpha


def _stratified_bootstrap(
    y: np.ndarray,
    prediction: np.ndarray,
    dummy_prediction: np.ndarray,
    label: np.ndarray,
    n_bootstrap: int,
) -> dict:
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(label == value) for value in np.unique(label)]
    r2_samples = []
    mse_gain_samples = []
    correlation_samples = []
    for _ in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        rng.shuffle(sampled)
        yt = y[sampled]
        yp = prediction[sampled]
        yd = dummy_prediction[sampled]
        if np.var(yt) <= 0:
            continue
        r2_samples.append(r2_score(yt, yp))
        mse_gain_samples.append(np.mean((yt - yd) ** 2) - np.mean((yt - yp) ** 2))
        correlation_samples.append(np.corrcoef(yt, yp)[0, 1])

    def interval(values) -> list[float]:
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    return {
        "r2_ci95": interval(r2_samples),
        "mse_improvement_vs_dummy_ci95": interval(mse_gain_samples),
        "pearson_r_ci95": interval(correlation_samples),
        "n_bootstrap": len(r2_samples),
    }


def analyze(cohort: str, n_bootstrap: int) -> Path:
    target = _load_target(cohort)
    aperiodic = _load_aperiodic_target(cohort)
    embeddings = LOADERS[cohort]()
    target_index = {
        serial: index for index, serial in enumerate(target["serial"])
    }
    output = {
        "meta": {
            "cohort": cohort,
            "target": (
                "mean 19-channel alpha-envelope DFA v2; 240 s; FIR 8-13 Hz; "
                "2 s to <=10% duration; 50% overlap"
            ),
            "primary_readout": (
                "5-fold OOF StandardScaler->PCA50->Ridge; all preprocessing "
                "fit on each outer training fold; alpha selected by efficient "
                "leave-one-out RidgeCV within the outer training fold"
            ),
            "descriptive_reference_r2": DESCRIPTIVE_REFERENCE_R2,
            "reference_timing": (
                "adopted during the post-v1 revision; not a prospectively "
                "prespecified confirmatory threshold"
            ),
            "interpretation": (
                "report point estimates, uncertainty, and cross-cohort "
                "replication; do not treat the descriptive reference as an "
                "inferential decision boundary"
            ),
            "bootstrap_interval_scope": (
                "diagnosis-stratified resampling of fixed OOF target/prediction "
                "pairs; overlapping training folds and the full pipeline are "
                "not refitted, so intervals are conditional rather than "
                "full-pipeline generalization intervals"
            ),
            "status": "primary encoding analysis",
        },
        "models": {},
        "aperiodic_comparison": {
            "target": "fixed-mode specparam aperiodic exponent, 1-40 Hz",
            "models": {},
        },
    }
    prediction_rows = {}

    for model, values in embeddings.items():
        serials = [
            serial
            for serial in target["serial"]
            if serial in values
            and target["eligible"][target_index[serial]]
            and np.isfinite(target["target"][target_index[serial]])
        ]
        if len(serials) < 25:
            output["models"][model] = {
                "status": "not analyzed",
                "reason": f"only {len(serials)} eligible matched subjects",
            }
            continue
        indices = np.asarray([target_index[serial] for serial in serials])
        x = np.vstack([values[serial] for serial in serials])
        y = target["target"][indices]
        label = target["label"][indices]
        prediction, dummy_prediction, alpha = _nested_oof_predictions(x, y)
        boot = _stratified_bootstrap(
            y,
            prediction,
            dummy_prediction,
            label,
            n_bootstrap,
        )
        r2 = float(r2_score(y, prediction))
        correlation = float(np.corrcoef(y, prediction)[0, 1])
        mse_gain = float(
            np.mean((y - dummy_prediction) ** 2)
            - np.mean((y - prediction) ** 2)
        )
        output["models"][model] = {
            "n": len(serials),
            "dimension": int(x.shape[1]),
            "r2": r2,
            "pearson_r": correlation,
            "mse_improvement_vs_dummy": mse_gain,
            "selected_alpha_by_outer_fold": alpha,
            **boot,
            "conditional_ci_upper_below_descriptive_reference": (
                boot["r2_ci95"][1] < DESCRIPTIVE_REFERENCE_R2
            ),
        }
        prediction_rows[model] = {
            "serial": serials,
            "target": y.tolist(),
            "prediction": prediction.tolist(),
            "dummy_prediction": dummy_prediction.tolist(),
            "label": label.tolist(),
        }
        print(
            f"[{cohort}] {model:10s} n={len(y):3d} "
            f"R2={r2:+.3f} CI={boot['r2_ci95']} r={correlation:+.3f}",
            flush=True,
        )

    for model, values in embeddings.items():
        serials = [
            serial
            for serial in target["serial"]
            if serial in values
            and serial in aperiodic
            and target["eligible"][target_index[serial]]
            and np.isfinite(aperiodic[serial])
        ]
        if len(serials) < 25:
            output["aperiodic_comparison"]["models"][model] = {
                "status": "not analyzed",
                "reason": f"only {len(serials)} eligible matched subjects",
            }
            continue
        indices = np.asarray([target_index[serial] for serial in serials])
        x = np.vstack([values[serial] for serial in serials])
        y = np.asarray([aperiodic[serial] for serial in serials])
        label = target["label"][indices]
        prediction, dummy_prediction, alpha = _nested_oof_predictions(x, y)
        boot = _stratified_bootstrap(
            y,
            prediction,
            dummy_prediction,
            label,
            n_bootstrap,
        )
        r2 = float(r2_score(y, prediction))
        output["aperiodic_comparison"]["models"][model] = {
            "n": len(serials),
            "dimension": int(x.shape[1]),
            "r2": r2,
            "pearson_r": float(np.corrcoef(y, prediction)[0, 1]),
            "selected_alpha_by_outer_fold": alpha,
            **boot,
        }
        print(
            f"[{cohort}] {model:10s} 1/f n={len(y):3d} "
            f"R2={r2:+.3f} CI={boot['r2_ci95']}",
            flush=True,
        )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULT_ROOT / f"primary_encoding_v2_{cohort}.json"
    output["oof_predictions"] = prediction_rows
    path.write_text(json.dumps(output, indent=2))
    print(f"wrote {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=list(LOADERS), required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    os.chdir(ROOT)
    analyze(args.cohort, args.n_bootstrap)


if __name__ == "__main__":
    main()
