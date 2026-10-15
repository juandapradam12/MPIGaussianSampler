# Case study: parallel Gaussian sampling with MPI

## Problem

Drawing large independent samples from a normal distribution is a common Monte Carlo building block. On a single core the cost scales as $O(N)$. With $P$ ranks the same workload can be partitioned so each process draws roughly $N / P$ values and a single gather reconstructs the full sample.

This repository implements that workflow end to end: a C/MPI sampler, a small Box–Muller module, and a Python plot that checks the empirical density against the closed-form PDF.

## Model

Samples are drawn from

$$
X \sim \mathcal{N}(\mu, \sigma^2)
$$

with density

$$
f(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).
$$

Uniforms on $(-1, 1)$ are transformed with the Marsaglia polar form of Box–Muller. Given $U, V$ such that $S = U^2 + V^2 \in (0, 1)$,

$$
Z_1 = U\sqrt{\frac{-2\ln S}{S}}, \qquad Z_2 = V\sqrt{\frac{-2\ln S}{S}}
$$

are i.i.d. standard normal, and $X = \mu + \sigma Z$ recovers the target law.

## Parallel design

| Stage | Responsibility |
| --- | --- |
| Partition | Rank $r$ owns $n_r = \lfloor N / P \rfloor$ or $\lfloor N / P \rfloor + 1$ samples |
| Seed | Each rank seeds independently (`time + rank · prime`) so streams do not collide |
| Draw | Local Box–Muller loop fills a contiguous buffer |
| Gather | `MPI_Gatherv` collects variable-sized chunks on rank 0 |
| Persist | Rank 0 writes one sample file; other ranks exit cleanly |

Point-to-point hello-world and send/recv programs under `src/examples/` isolate the MPI primitives used by the sampler.

## Validation

After a run, the histogram of the gathered sample should track $f(x \mid \mu, \sigma)$. The plotting script overlays both and writes `output/distribution.png`. For large $N$ the match is visual; mean and variance of the file should also approach $\mu$ and $\sigma^2$.

## Design notes

- **Separation of concerns.** RNG logic lives in `box_muller.c`; communication and I/O live in `sampler.c`; visualization is a separate Python tool.
- **Correct MPI usage.** The original exercise wrote from every rank into the same path. The rewrite gathers once and writes once.
- **Portable CLI.** Arguments are validated on rank 0, broadcast, and shared so every process agrees on $N$, $\mu$, and $\sigma$.
- **Reproducible layout.** `Makefile` targets (`all`, `run`, `plot`, `demo`, `examples`) encode the intended workflow without cluster-specific scripts.
