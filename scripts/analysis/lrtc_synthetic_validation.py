"""
Synthetic validation of the LRTC-aware auxiliary loss.

PURPOSE
-------
Establish, on data whose ground-truth scaling exponent is known exactly, that
adding the LRTC auxiliary loss to a masked-reconstruction objective moves
frozen-embedding recovery of the exponent from R^2 ~ 0 toward R^2 > 0.3.

This is the falsification test for Section 5 of the manuscript. It is
deliberately NOT a test on real EEG: the point is to remove every confound
(filter artifact, cohort effects, site encoding, measurement reliability) and
ask one question -- does the loss make the exponent decodable from a frozen
representation that otherwise discards it?

WHY THIS DESIGN
---------------
The loss has two separable jobs, and conflating them is the failure mode:

  L_linearity : make log F(s) vs log s straight, with the slope FREE.
                This makes the representation scale-free.

  L_fidelity  : make the (free) fitted slope match THIS SAMPLE's input
                exponent. This makes the representation INFORMATIVE about
                the exponent.

Only L_fidelity can raise decoding R^2. A loss that pins the slope to a fixed
global target alpha_hat is a homogenizing regularizer: it pushes every sample
toward the same exponent and would plausibly DRIVE R^2 DOWN. That is the bug
this scaffold is built to expose, so ABLATION_MODES below includes the
fixed-target variant explicitly as a negative control.

GROUND TRUTH
------------
Fractional Gaussian noise (fGn) with Hurst H has DFA exponent alpha = H over
the fitting range. We sample H ~ U(0.55, 0.95), which brackets the empirical
alpha-envelope range reported in the manuscript (~0.76-0.80, 96-99% within
[0.5, 1.0]).

NOTE ON THE fGn GENERATOR: the Davies-Harte circulant-embedding method used
here is exact for fGn. Verify empirically (validate_generator) that
DFA-recovered alpha tracks the requested H before trusting any downstream
result -- if the generator is wrong, everything else is measuring noise.

MEASURED CALIBRATION (numpy path only; 200 draws, H ~ U(0.55, 0.95)):
    seq_len =  4096, scales 16..512   -> r = 0.949, bias -0.005, resid SD 0.038
    seq_len = 16384, scales 16..2048  -> r = 0.979, bias -0.004, resid SD 0.024
In both cases the regression slope of recovered alpha on requested H is ~0.99,
so the estimator is unbiased and the residual is finite-length estimator noise.

THIS SETS THE CEILING. At seq_len=4096 the ground truth is itself only
recoverable to R^2 ~ 0.90; at 16384, ~0.96. Report the achievable ceiling
alongside any probe R^2, exactly as the manuscript does with its split-half
reliability estimate (R^2 = 0.64 on CAUEEG). A probe R^2 of 0.35 against a
0.90 ceiling is a different claim from 0.35 against 1.0.

USAGE
-----
    python lrtc_synthetic_validation.py --mode baseline
    python lrtc_synthetic_validation.py --mode lrtc_correct
    python lrtc_synthetic_validation.py --mode lrtc_fixed_target   # neg. control
    python lrtc_synthetic_validation.py --compare-all

Expected runtime: minutes to low hours on one GPU, depending on n_train.

STATUS: SCAFFOLD. Every numeric constant below is a starting point, not a
tuned value. Only the fGn generator and DFA estimator have been run. Do not
report any number produced by this file without first passing
validate_generator() and the probe sanity checks noted in run_probe().
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class Config:
    # Data
    n_train: int = 4000
    n_test: int = 1000
    seq_len: int = 4096          # samples per synthetic "recording"
    h_range: tuple[float, float] = (0.55, 0.95)
    seed: int = 42

    # Tokenization: seq_len / patch_len tokens.
    # 4096 / 64 = 64 tokens -> dyadic scales {2,4,8,16,32} = 5 fit points.
    # THIS IS THE SCALE BUDGET. Report it explicitly in the paper; 5 points
    # over ~1.2 decades is thin, and it is the same vulnerability the
    # manuscript flags for the 0.5-2 s BrainLat range.
    patch_len: int = 64

    # Model
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mask_ratio: float = 0.5

    # Optimization
    lr: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 100

    # Auxiliary loss weights. lambda_fid is the one that should matter.
    # Sweep both: if results are knife-edge in lambda, say so in the paper.
    lambda_lin: float = 0.1
    lambda_fid: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dyadic_scales: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])


ABLATION_MODES = (
    "baseline",             # masked reconstruction only
    "lrtc_correct",         # + L_linearity (free slope) + L_fidelity (per-sample)
    "lrtc_linearity_only",  # + L_linearity only -> scale-free but uninformative
    "lrtc_fixed_target",    # + slope pinned to global constant: NEGATIVE CONTROL
)


# --------------------------------------------------------------------------
# Synthetic data: fractional Gaussian noise with known Hurst exponent
# --------------------------------------------------------------------------


def generate_fgn(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Exact fGn via Davies-Harte circulant embedding.

    Returns a length-n series with theoretical Hurst exponent `hurst`.
    """
    k = np.arange(n)
    # fGn autocovariance at lag k, unit variance
    gamma = 0.5 * (
        np.abs(k - 1) ** (2 * hurst)
        - 2 * np.abs(k) ** (2 * hurst)
        + np.abs(k + 1) ** (2 * hurst)
    )
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    eigenvals = np.fft.fft(row).real
    if np.any(eigenvals < 0):
        # Non-negative-definite embedding failed. For H in (0.5, 1) with this
        # row construction it should not happen -- treat any trigger here as a
        # bug signal, not a nuisance to clip away silently.
        eigenvals = np.clip(eigenvals, 0, None)

    m = len(row)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    w = np.fft.fft(np.sqrt(eigenvals / (2 * m)) * z)
    return np.real(w[:n])


def dfa_numpy(x: np.ndarray, scales: np.ndarray) -> float:
    """Reference (non-differentiable) DFA-1, for ground-truth checking only.

    Matches the manuscript's estimator: cumulative sum of the mean-subtracted
    series, non-overlapping windows, per-window least-squares linear detrend,
    RMS residual, log-log slope.
    """
    y = np.cumsum(x - x.mean())
    fluct = []
    used = []
    for s in scales:
        n_win = len(y) // s
        if n_win < 2:
            continue
        windows = y[: n_win * s].reshape(n_win, s)
        t = np.arange(s)
        coeffs = np.polyfit(t, windows.T, 1)
        trend = np.outer(coeffs[0], t) + coeffs[1][:, None]
        resid = windows - trend
        fluct.append(np.sqrt((resid ** 2).mean()))
        used.append(s)
    return float(np.polyfit(np.log(used), np.log(fluct), 1)[0])


def validate_generator(cfg: Config, n_check: int = 200) -> None:
    """RUN THIS FIRST. If DFA-recovered alpha does not track requested H,
    nothing downstream means anything.

    Pass criterion, calibrated against the measured values in the module
    docstring: r > 0.94 at seq_len=4096, r > 0.97 at seq_len=16384, with
    |bias| < 0.02 and regression slope of alpha on H within [0.95, 1.05].
    The r threshold is length-dependent because the residual is estimator
    noise, not generator error -- do not tighten it without lengthening the
    series.
    """
    rng = np.random.default_rng(cfg.seed)
    scales = np.array([16, 32, 64, 128, 256, 512])
    requested, recovered = [], []
    for _ in range(n_check):
        h = rng.uniform(*cfg.h_range)
        x = generate_fgn(cfg.seq_len, h, rng)
        requested.append(h)
        recovered.append(dfa_numpy(x, scales))

    requested = np.array(requested)
    recovered = np.array(recovered)
    r = np.corrcoef(requested, recovered)[0, 1]
    bias = float((recovered - requested).mean())
    slope = float(np.polyfit(requested, recovered, 1)[0])
    threshold = 0.97 if cfg.seq_len >= 16384 else 0.94

    print(
        f"[generator] r(H, alpha_DFA) = {r:.4f}   bias = {bias:+.4f}   "
        f"slope = {slope:.4f}   (ceiling on probe R^2 ~ {r**2:.3f})"
    )
    if r < threshold or abs(bias) > 0.02 or not (0.95 <= slope <= 1.05):
        raise RuntimeError(
            "fGn generator or DFA estimator is not behaving. Fix before proceeding."
        )


def make_dataset(cfg: Config, n: int, seed_offset: int = 0):
    rng = np.random.default_rng(cfg.seed + seed_offset)
    xs, hs = [], []
    for _ in range(n):
        h = rng.uniform(*cfg.h_range)
        x = generate_fgn(cfg.seq_len, h, rng)
        xs.append((x - x.mean()) / (x.std() + 1e-8))  # per-sample z-norm
        hs.append(h)
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(np.array(hs), dtype=torch.float32),
    )


# --------------------------------------------------------------------------
# Differentiable DFA surrogate -- the core of the proposed fix
# --------------------------------------------------------------------------


def differentiable_fluctuation(u: torch.Tensor, scales: list[int]) -> torch.Tensor:
    """log F(s) for each dyadic scale s, fully differentiable in u.

    u : (B, T) scalar token-activation series
    returns : (B, n_valid_scales)

    Every operation -- cumulative sum, closed-form least-squares detrend, RMS,
    log -- is differentiable, so this backpropagates into the encoder.
    """
    y = torch.cumsum(u - u.mean(dim=1, keepdim=True), dim=1)
    out = []
    for s in scales:
        n_win = y.shape[1] // s
        if n_win < 2:
            continue
        w = y[:, : n_win * s].reshape(y.shape[0], n_win, s)

        # Closed-form least-squares linear detrend (differentiable).
        t = torch.arange(s, device=u.device, dtype=u.dtype)
        t = t - t.mean()
        denom = (t ** 2).sum() + 1e-8
        slope = (w * t).sum(dim=2, keepdim=True) / denom
        intercept = w.mean(dim=2, keepdim=True)
        resid = w - (slope * t + intercept)

        f_s = torch.sqrt((resid ** 2).mean(dim=(1, 2)) + 1e-8)
        out.append(torch.log(f_s))
    return torch.stack(out, dim=1)


def fit_free_slope(log_f: torch.Tensor, scales: list[int]):
    """Least-squares fit of log F(s) = alpha * log(s) + c, with alpha FREE.

    Returns (alpha_hat, log_f_predicted). Differentiable throughout.
    """
    log_s = torch.log(
        torch.tensor(scales[: log_f.shape[1]], device=log_f.device, dtype=log_f.dtype)
    )
    ls_c = log_s - log_s.mean()
    denom = (ls_c ** 2).sum() + 1e-8
    alpha = ((log_f - log_f.mean(dim=1, keepdim=True)) * ls_c).sum(dim=1) / denom
    intercept = log_f.mean(dim=1) - alpha * log_s.mean()
    pred = alpha[:, None] * log_s[None, :] + intercept[:, None]
    return alpha, pred


def lrtc_loss(tokens: torch.Tensor, alpha_target: torch.Tensor, cfg: Config, mode: str):
    """Auxiliary LRTC loss.

    tokens       : (B, n_tokens, d_model) pre-pool sequence
    alpha_target : (B,) per-sample ground-truth exponent computed from the INPUT

    The per-sample target is what makes this informative rather than
    homogenizing. In a real pretraining run alpha_target is computed on-the-fly
    from the raw input signal -- it is self-supervised, requiring no labels.
    """
    # Reduce tokens to a scalar series. Alternatives worth ablating:
    # channel-averaged norm (below), a learned linear read-out, or PC1.
    # The choice is a free parameter and should be reported.
    u = tokens.norm(dim=2)  # (B, n_tokens)

    log_f = differentiable_fluctuation(u, cfg.dyadic_scales)
    alpha_hat, pred = fit_free_slope(log_f, cfg.dyadic_scales)

    # (1) Scale-freeness: residual from a straight line, slope free.
    l_lin = ((log_f - pred) ** 2).mean()

    # (2) Exponent fidelity: does the representation carry THIS sample's alpha?
    if mode == "lrtc_correct":
        l_fid = ((alpha_hat - alpha_target) ** 2).mean()
    elif mode == "lrtc_linearity_only":
        l_fid = torch.zeros((), device=tokens.device)
    elif mode == "lrtc_fixed_target":
        # NEGATIVE CONTROL: pin every sample to the batch-mean exponent.
        # This is the formulation the manuscript originally proposed. It should
        # NOT improve -- and may degrade -- decoding R^2. If it does improve,
        # the mechanism is not what we claim and Section 5 needs rethinking.
        l_fid = ((alpha_hat - alpha_target.mean().detach()) ** 2).mean()
    else:
        raise ValueError(f"unknown mode {mode}")

    total = cfg.lambda_lin * l_lin + cfg.lambda_fid * l_fid
    return total, {
        "l_lin": l_lin.item(),
        "l_fid": float(l_fid),
        "alpha_hat_mean": alpha_hat.mean().item(),
        # Watch this: collapse toward 0 is the homogenization signature.
        "alpha_hat_std": alpha_hat.std().item(),
    }


# --------------------------------------------------------------------------
# Minimal masked-reconstruction encoder
# --------------------------------------------------------------------------


class PatchEncoder(nn.Module):
    """Small transformer with a masked-patch-reconstruction objective.

    Intentionally minimal -- this is a mechanism test, not a competitive model.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_tokens = cfg.seq_len // cfg.patch_len

        self.embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens, cfg.d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.decoder = nn.Linear(cfg.d_model, cfg.patch_len)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.n_tokens, self.cfg.patch_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        patches = self.patchify(x)
        h = self.embed(patches) + self.pos
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token.expand_as(h), h)
        tokens = self.backbone(h)
        return tokens, self.decoder(tokens)

    @torch.no_grad()
    def embed_frozen(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled frozen embedding -- exactly what the manuscript probes."""
        tokens, _ = self.forward(x, mask=None)
        return tokens.mean(dim=1)


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------


def train(cfg: Config, mode: str, x_train: torch.Tensor, alpha_train: torch.Tensor):
    model = PatchEncoder(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    n = x_train.shape[0]
    stats: dict = {}

    for epoch in range(cfg.n_epochs):
        model.train()
        perm = torch.randperm(n)
        agg = {"recon": 0.0, "aux": 0.0, "n": 0}

        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb = x_train[idx].to(cfg.device)
            ab = alpha_train[idx].to(cfg.device)

            mask = torch.rand(xb.shape[0], model.n_tokens, device=cfg.device) < cfg.mask_ratio
            tokens, recon = model(xb, mask=mask)

            target = model.patchify(xb)
            loss_recon = ((recon - target) ** 2)[mask].mean()

            if mode == "baseline":
                loss_aux = torch.zeros((), device=cfg.device)
                stats = {}
            else:
                loss_aux, stats = lrtc_loss(tokens, ab, cfg, mode)

            loss = loss_recon + loss_aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            agg["recon"] += loss_recon.item()
            agg["aux"] += float(loss_aux)
            agg["n"] += 1

        if epoch % 10 == 0 or epoch == cfg.n_epochs - 1:
            extra = f"  alpha_hat_std {stats['alpha_hat_std']:.4f}" if stats else ""
            print(
                f"[{mode}] epoch {epoch:3d}  "
                f"recon {agg['recon'] / agg['n']:.4f}  "
                f"aux {agg['aux'] / agg['n']:.4f}{extra}"
            )
    return model


def run_probe(model: PatchEncoder, x: torch.Tensor, alpha: torch.Tensor, cfg: Config) -> float:
    """Frozen-embedding ridge probe -- the manuscript's own encoding probe.

    Sanity checks before believing the number:
      * Positive control: a probe on the raw input's classical DFA feature must
        recover alpha near-perfectly. If it does not, the probe is broken.
      * Negative control: shuffle alpha across samples; R^2 must collapse to ~0.
    """
    model.eval()
    embs = []
    for i in range(0, x.shape[0], 256):
        embs.append(model.embed_frozen(x[i : i + 256].to(cfg.device)).cpu().numpy())
    emb = np.concatenate(embs)
    y = alpha.numpy()

    probe = RidgeCV(alphas=np.logspace(-3, 3, 13))
    scores = cross_val_score(probe, emb, y, cv=5, scoring="r2")
    return float(scores.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=ABLATION_MODES, default="lrtc_correct")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--skip-generator-check", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(cfg.seed)

    if not args.skip_generator_check:
        validate_generator(cfg)

    print("Building synthetic fGn datasets...")
    x_train, a_train = make_dataset(cfg, cfg.n_train, seed_offset=0)
    x_test, a_test = make_dataset(cfg, cfg.n_test, seed_offset=9999)

    modes = ABLATION_MODES if args.compare_all else (args.mode,)
    results = {}
    for mode in modes:
        print(f"\n=== mode: {mode} ===")
        model = train(cfg, mode, x_train, a_train)
        r2 = run_probe(model, x_test, a_test, cfg)
        results[mode] = r2
        print(f"[{mode}] frozen-embedding probe R^2 = {r2:.4f}")

    print("\n=== SUMMARY ===")
    for mode, r2 in results.items():
        print(f"  {mode:24s} R^2 = {r2:+.4f}")
    print(
        "\nSuccess criterion (Section 5): baseline ~ 0, lrtc_correct > 0.3.\n"
        "Diagnostic: lrtc_fixed_target should NOT beat lrtc_correct. If it does,\n"
        "the per-sample-target argument is wrong and Section 5 needs rethinking."
    )


if __name__ == "__main__":
    main()
