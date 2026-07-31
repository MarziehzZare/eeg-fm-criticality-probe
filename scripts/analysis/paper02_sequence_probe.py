"""Paper-02-specific order-aware sequence probe.

This module deliberately does not modify ``reve_prepool_token_probe.py``.  That
legacy helper produced governed cross-population artifacts under its historical
training loop.  Paper 02 needs the corrected loop and explicit batch-size
sensitivity, so the behavior is isolated here instead of silently changing
other analyses.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.seed import set_seed

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ConvProbe(nn.Module):
    """Small order-aware 1D CNN over the token/epoch axis."""

    def __init__(self, d: int, h: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, h, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(h, h, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    epochs: int = 200,
    patience: int = 20,
) -> tuple[ConvProbe, int, float]:
    """Train one fold with one complete subject permutation per epoch."""
    set_seed(seed)
    model = ConvProbe(d=x_train.shape[1]).to(DEV)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(x_train, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y_train, dtype=torch.float32, device=DEV)
    xv = torch.tensor(x_val, dtype=torch.float32, device=DEV)
    yv = torch.tensor(y_val, dtype=torch.float32, device=DEV)
    best, best_state, bad, epochs_run = 1e9, None, 0, 0
    n = len(xt)
    effective_batch = min(batch_size, n)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        epochs_run += 1
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, effective_batch):
            idx = torch.as_tensor(
                order[start : start + effective_batch],
                dtype=torch.long,
                device=DEV,
            )
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        if val_loss < best - 1e-4:
            best = val_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, epochs_run, best


def cv_r2(
    sequences: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    config_seed: int,
    shuffle_seed: int | None = None,
) -> tuple[float, float, float, list[dict[str, float | int]]]:
    """Return 5-fold OOF R2 with explicit split/init and order seeds."""
    x = sequences.astype(np.float32).copy()
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        for subject_idx in range(len(x)):
            x[subject_idx] = x[subject_idx][rng.permutation(x.shape[1])]
    x = np.transpose(x, (0, 2, 1))

    prediction = np.zeros(len(y))
    epochs_run: list[int] = []
    val_losses: list[float] = []
    per_fold: list[dict[str, float | int]] = []
    splitter = KFold(5, shuffle=True, random_state=42 + config_seed)
    for train_idx, test_idx in splitter.split(x):
        cut = max(8, len(train_idx) // 6)
        val_idx, fit_idx = train_idx[:cut], train_idx[cut:]
        mu = x[fit_idx].mean((0, 2), keepdims=True)
        sd = x[fit_idx].std((0, 2), keepdims=True) + 1e-6
        y_mu = y[fit_idx].mean()
        y_sd = y[fit_idx].std() + 1e-6
        xn = (x - mu) / sd
        model, ran, best = train_fold(
            xn[fit_idx],
            (y[fit_idx] - y_mu) / y_sd,
            xn[val_idx],
            (y[val_idx] - y_mu) / y_sd,
            batch_size=batch_size,
            seed=config_seed,
        )
        epochs_run.append(ran)
        val_losses.append(best)
        per_fold.append({"epochs": int(ran), "best_val_loss": round(float(best), 5)})
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(xn[test_idx], dtype=torch.float32, device=DEV))
        prediction[test_idx] = pred.cpu().numpy() * y_sd + y_mu

    return (
        float(r2_score(y, prediction)),
        float(np.mean(epochs_run)),
        float(np.mean(val_losses)),
        per_fold,
    )


def seeded_cv_r2(
    sequences: np.ndarray,
    y: np.ndarray,
    *,
    shuffle_order: bool,
    seed: int,
    batch_size: int,
) -> float:
    """Compatibility wrapper for the superseded repeated-seed diagnostic."""
    shuffle_seed = 123 + seed if shuffle_order else None
    score, _, _, _ = cv_r2(
        sequences,
        y,
        batch_size=batch_size,
        config_seed=seed,
        shuffle_seed=shuffle_seed,
    )
    return score


def ridge_cv_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Five-fold OOF ridge R2 used for pooled and classical comparators."""
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    prediction = cross_val_predict(
        pipe,
        x,
        y,
        cv=KFold(5, shuffle=True, random_state=42),
    )
    return float(r2_score(y, prediction))
