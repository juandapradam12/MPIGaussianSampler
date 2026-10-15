#!/usr/bin/env python3
"""Compare an empirical sample against the N(mu, sigma^2) density."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


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
    if samples.size == 0:
        raise SystemExit(f"No samples found in {args.input}")

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
