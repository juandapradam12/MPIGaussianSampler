#!/usr/bin/env python3
"""Compare an empirical sample against the N(mu, sigma^2) density."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def normal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - mu) / (sigma * math.sqrt(2.0))
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(z))


def chi2_sf(dof: int, stat: float) -> float:
    """Upper-tail P(X >= stat) for X ~ chi-square(dof)."""
    if dof <= 0 or stat < 0.0 or math.isnan(stat):
        return float("nan")
    # Q(k, x) regularized upper gamma with k = dof/2, x = stat/2.
    k = 0.5 * dof
    x = 0.5 * stat
    if x == 0.0:
        return 1.0
    # Series for lower gamma P(k,x), then Q = 1 - P (stable for moderate x).
    term = math.exp(-x + k * math.log(x) - math.lgamma(k)) / k
    lower = term
    for n in range(1, 10_000):
        term *= x / (k + n)
        lower += term
        if abs(term) < abs(lower) * 1e-12:
            break
    return float(max(0.0, min(1.0, 1.0 - lower)))


def chi_square_gof(
    samples: np.ndarray, mu: float, sigma: float, n_bins: int
) -> tuple[float, int, float]:
    """Pearson chi-square GOF vs N(mu, sigma^2); returns (stat, dof, p-value)."""
    lo = mu - 4.0 * sigma
    hi = mu + 4.0 * sigma
    edges = np.linspace(lo, hi, n_bins + 1)
    observed, _ = np.histogram(samples, bins=edges)

    # Fold tails outside [lo, hi] into the end bins.
    observed = observed.astype(float)
    observed[0] += np.sum(samples < lo)
    observed[-1] += np.sum(samples > hi)

    expected = samples.size * np.diff(normal_cdf(edges, mu, sigma))

    # Merge bins with small expected counts from the left.
    obs_m: list[float] = []
    exp_m: list[float] = []
    o_acc = 0.0
    e_acc = 0.0
    for o, e in zip(observed, expected):
        o_acc += o
        e_acc += e
        if e_acc >= 5.0:
            obs_m.append(o_acc)
            exp_m.append(e_acc)
            o_acc = 0.0
            e_acc = 0.0
    if e_acc > 0.0:
        if exp_m:
            obs_m[-1] += o_acc
            exp_m[-1] += e_acc
        else:
            obs_m.append(o_acc)
            exp_m.append(e_acc)

    obs_arr = np.asarray(obs_m)
    exp_arr = np.asarray(exp_m)
    if exp_arr.size < 2 or np.any(exp_arr <= 0.0):
        return float("nan"), 0, float("nan")

    stat = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
    dof = int(exp_arr.size - 1)
    return stat, dof, chi2_sf(dof, stat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a histogram of sampled values against the theoretical PDF."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/sample.dat"),
        help="Path to the sample file written by the MPI sampler",
    )
    parser.add_argument("--mu", type=float, required=True, help="Gaussian mean")
    parser.add_argument(
        "--sigma", type=float, required=True, help="Gaussian standard deviation"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins (default: 80)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/distribution.png"),
        help="Path for the saved figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sigma <= 0.0:
        raise SystemExit("--sigma must be positive")

    samples = np.loadtxt(args.input)
    if samples.ndim != 1:
        samples = np.ravel(samples)
    if samples.size == 0:
        raise SystemExit(f"No samples found in {args.input}")

    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    mean_err = abs(mean - args.mu)
    var_err = abs(std**2 - args.sigma**2)
    chi2, dof, p_value = chi_square_gof(samples, args.mu, args.sigma, min(args.bins, 40))

    print(f"n              = {samples.size}")
    print(f"sample mean    = {mean:.6f}  (target {args.mu:g}, |err|={mean_err:.6f})")
    print(
        f"sample std     = {std:.6f}  (target {args.sigma:g}, "
        f"|var err|={var_err:.6f})"
    )
    if dof > 0 and not math.isnan(chi2):
        print(f"chi-square GOF = {chi2:.3f} on {dof} dof (p={p_value:.4f})")
    else:
        print("chi-square GOF = n/a (insufficient bin mass)")

    x_min = min(samples.min(), args.mu - 4.0 * args.sigma)
    x_max = max(samples.max(), args.mu + 4.0 * args.sigma)
    x = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        samples,
        bins=args.bins,
        density=True,
        color="#4C78A8",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
        label="Empirical sample",
    )
    ax.plot(
        x,
        gaussian_pdf(x, args.mu, args.sigma),
        color="#E45756",
        linewidth=2.0,
        label=rf"PDF $\mathcal{{N}}({args.mu:g}, {args.sigma:g}^2)$",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Parallel Box–Muller sample vs. theoretical Gaussian")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
