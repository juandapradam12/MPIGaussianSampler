#include "box_muller.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Parallel Monte Carlo sampler for N(mu, sigma^2).
 *
 * Each MPI rank draws a contiguous share of the sample, then the root
 * process gathers the full vector and writes it to disk.
 *
 * Usage:
 *   mpiexec -n <P> ./bin/sampler <N> <mu> <sigma> [output_path]
 */

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s <N> <mu> <sigma> [output_path]\n"
            "  N            total number of samples (positive integer)\n"
            "  mu           mean of the Gaussian\n"
            "  sigma        standard deviation (must be > 0)\n"
            "  output_path  optional path for the sample file "
            "(default: output/sample.dat)\n",
            prog);
}

static int parse_args(int argc, char **argv, long *n_total, double *mu,
                      double *sigma, const char **output_path)
{
    if (argc < 4 || argc > 5) {
        return -1;
    }

    char *end = NULL;
    *n_total = strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || *n_total <= 0) {
        return -1;
    }

    *mu = strtod(argv[2], &end);
    if (end == argv[2] || *end != '\0') {
        return -1;
    }

    *sigma = strtod(argv[3], &end);
    if (end == argv[3] || *end != '\0' || *sigma <= 0.0) {
        return -1;
    }

    *output_path = (argc == 5) ? argv[4] : "output/sample.dat";
    return 0;
}

static void local_counts(long n_total, int size, int rank, int *local_n,
                         int *counts, int *displs)
{
    const int base = (int)(n_total / size);
    const int rem = (int)(n_total % size);

    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }

    *local_n = counts[rank];
}

static int write_samples(const char *path, const double *values, long n)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        perror(path);
        return -1;
    }

    for (long i = 0; i < n; ++i) {
        if (fprintf(fp, "%.10f\n", values[i]) < 0) {
            fclose(fp);
            return -1;
        }
    }

    if (fclose(fp) != 0) {
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n_total = 0;
    double mu = 0.0;
    double sigma = 0.0;
    const char *output_path = NULL;

    int parse_ok = 0;
    if (rank == 0) {
        parse_ok = (parse_args(argc, argv, &n_total, &mu, &sigma, &output_path) == 0);
        if (!parse_ok) {
            usage(argv[0]);
        }
    }

    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parse_ok) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Bcast(&n_total, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *counts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    if (counts == NULL || displs == NULL) {
        fprintf(stderr, "rank %d: failed to allocate count buffers\n", rank);
        free(counts);
        free(displs);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int local_n = 0;
    local_counts(n_total, size, rank, &local_n, counts, displs);

    /* Distinct stream per rank; still reproducible for a fixed process count. */
    srand((unsigned int)(time(NULL) + rank * 9973));

    double *local = (double *)malloc((size_t)local_n * sizeof(double));
    if (local == NULL) {
        fprintf(stderr, "rank %d: failed to allocate local sample buffer\n", rank);
        free(counts);
        free(displs);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < local_n; ++i) {
        local[i] = box_muller_normal(mu, sigma);
    }

    double *global = NULL;
    if (rank == 0) {
        global = (double *)malloc((size_t)n_total * sizeof(double));
        if (global == NULL) {
            fprintf(stderr, "rank 0: failed to allocate global sample buffer\n");
            free(local);
            free(counts);
            free(displs);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Gatherv(local, local_n, MPI_DOUBLE, global, counts, displs, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    int write_status = 0;
    if (rank == 0) {
        write_status = write_samples(output_path, global, n_total);
        if (write_status == 0) {
            printf("Wrote %ld samples from N(%.6f, %.6f^2) using %d ranks to %s\n",
                   n_total, mu, sigma, size, output_path);
        } else {
            fprintf(stderr, "Failed to write samples to %s\n", output_path);
        }
        free(global);
    }

    free(local);
    free(counts);
    free(displs);

    MPI_Bcast(&write_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return (write_status == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
