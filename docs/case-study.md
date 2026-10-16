# Case study: parallel Gaussian sampling with MPI

## Problem

Drawing large independent samples from a normal distribution is a common Monte Carlo building block. On a single core the cost scales as $O(N)$. With $P$ ranks the same workload can be partitioned so each process draws roughly $N / P$ values and a single gather reconstructs the full sample.

This repository implements that workflow end to end: a C/MPI sampler, a small Box–Muller module, and a Python tool that both plots and scores the empirical distribution against the closed-form PDF.

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
| Seed | Base seed $S$ (CLI `--seed`); rank $r$ uses $S + 9973\,r$ |
| Draw | Local Box–Muller loop fills a contiguous buffer |
| Gather | `MPI_Gatherv` collects variable-sized chunks on rank 0 |
| Persist | Rank 0 writes one sample file; other ranks exit cleanly |

Point-to-point hello-world and send/recv programs under `src/examples/` isolate the MPI primitives used by the sampler.

## Timing and scaling

The sampler reports the max-over-ranks wall time for the draw + gather phase (`MPI_Wtime`). For fixed $N$ (strong scaling), that time should drop as $P$ grows until communication or RNG overhead dominates. For fixed work per rank (weak scaling), wall time should stay roughly flat while $N \propto P$.

### Strong scaling ($N = 2\times 10^6$, seed $42$)

Measured on a single oversubscribed workstation (Open MPI `--oversubscribe`). Speedup is $T_1 / T_P$; efficiency is speedup $/\,P$.

| $P$ | Wall time (s) | Speedup | Efficiency |
| --- | ---: | ---: | ---: |
| 1 | 0.0702 | 1.00× | 100% |
| 2 | 0.0324 | 2.17× | 108% |
| 4 | 0.0355 | 1.98× | 49% |
| 8 | 0.0274 | 2.56× | 32% |

The $1\to 2$ step nearly doubles throughput. Beyond that, gather cost and core oversubscription flatten the curve — expected on a laptop/CI runner, and a useful reminder that MPI gains need real hardware parallelism.

Reproduce:

```bash
for P in 1 2 4 8; do
  mpiexec --oversubscribe -n $P ./bin/sampler 2000000 2.5 3.5 /tmp/s.dat --seed 42
done
```

## Validation

The plotting script reports:

- sample mean and standard deviation versus $(\mu, \sigma)$
- absolute mean error and absolute variance error
- a Pearson $\chi^2$ goodness-of-fit against $\mathcal{N}(\mu, \sigma^2)$ on merged histogram bins (expected count $\ge 5$), with a $p$-value from the upper incomplete gamma

Together with the PDF overlay in `output/distribution.png`, these checks confirm that the parallel sample matches the target law for large $N$.

## Design notes

- **Separation of concerns.** RNG logic lives in `box_muller.c`; communication, seeding, timing, and I/O live in `sampler.c`; visualization and GOF live in Python.
- **Single writer.** Local buffers are gathered once; only rank 0 touches the output path.
- **Shared configuration.** Arguments are validated on rank 0 and broadcast so every process agrees on $N$, $\mu$, $\sigma$, and $S$.
- **Reproducible workflow.** `Makefile` targets and GitHub Actions CI encode the intended build → sample → validate path.
