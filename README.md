# Parallel Gaussian Sampler (MPI)

Monte Carlo sampling of $\mathcal{N}(\mu, \sigma^2)$ distributed across MPI ranks, with quantitative checks against the theoretical density.

Portfolio case study: a small numerical workload with clear module boundaries, correct collective communication, reproducible seeding, and a build/plot/CI pipeline.

> Suggested GitHub repository name: `mpi-gaussian-sampler` (rename under **Settings → General** if the remote is still titled as a course exercise).

## What it does

1. Each MPI rank draws a share of $N$ independent normals via the Box–Muller (Marsaglia polar) transform.
2. Rank 0 gathers the local buffers with `MPI_Gatherv` and writes a single sample file.
3. A Python script overlays the empirical histogram on

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-(x-\mu)^2 / (2\sigma^2)}
$$

   and reports sample mean, variance error, and a Pearson $\chi^2$ goodness-of-fit summary.

## Layout

```text
src/
  box_muller.c          Box–Muller transform
  include/box_muller.h
  sampler.c             MPI driver (partition → draw → gather → write)
  examples/             Minimal hello-world and send/recv programs
scripts/
  plot_distribution.py  Histogram, PDF overlay, and GOF stats
docs/
  case-study.md         Problem statement, math, and design notes
.github/workflows/ci.yml
output/                 Generated samples and figures (gitignored)
```

## Requirements

- C compiler with MPI (`mpicc`, Open MPI or MPICH)
- Python 3 with `numpy` and `matplotlib` (`pip install -r requirements.txt`)

## Quick start

```bash
make demo
# equivalent to:
#   make
#   mpiexec -n 4 ./bin/sampler 100000 2.5 3.5 output/sample.dat --seed 42
#   python3 scripts/plot_distribution.py --mu 2.5 --sigma 3.5
```

Override defaults as needed:

```bash
make run N=500000 MU=0 SIGMA=1 NP=8 SEED=7
make plot MU=0 SIGMA=1
```

`make run` uses `mpiexec --oversubscribe` so the demo works on a single workstation. Pass `--seed` for bit-reproducible streams at a fixed process count (rank $r$ seeds with $S + 9973\,r$).

## Example result

![Empirical histogram vs. N(2.5, 3.5²) PDF](docs/assets/distribution.png)

Example programs:

```bash
make examples
mpiexec -n 2 ./bin/mpi_hello
mpiexec -n 2 ./bin/mpi_send_recv
```

## Design highlights

- Partitioned Monte Carlo with `MPI_Gatherv` and a single writer on rank 0
- Isolated Box–Muller module, MPI driver, and Python validation/plotting
- Reproducible `--seed`, wall-clock timing for the draw + gather phase
- Portable `Makefile` targets and GitHub Actions CI

## Further reading

See [docs/case-study.md](docs/case-study.md) for the sampling model, parallel partition, timing notes, and validation details.
