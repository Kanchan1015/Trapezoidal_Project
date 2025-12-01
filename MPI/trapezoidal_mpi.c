

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Function to integrate */
static double f(double x) {
    return sin(x);
}

/* Compute trapezoidal integral on [local_a, local_b] using local_n subdivisions of width h */
static double local_trap(double local_a, double local_b, long local_n, double h) {
    double sum;
    double x;
    long i;

    sum = (f(local_a) + f(local_b)) * 0.5;
    x = local_a;
    for (i = 1; i < local_n; ++i) {
        x += h;
        sum += f(x);
    }
    return sum * h;
}

int main(int argc, char *argv[]) {
    int rank, size;
    long n;
    double a = 0.0, b = M_PI;
    double h;
    long base, rem;
    long local_n;
    long offset;            /* number of subdivisions before this rank */
    double local_a, local_b;
    double local_result = 0.0, total_result = 0.0;
    struct timespec tstart, tend;
    double time_sec = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <n_subdivisions>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    n = atol(argv[1]);
    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "n must be a positive integer\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* compute per-process subdivision counts (handle remainder) */
    base = n / size;
    rem  = n % size;
    if (rank < rem) {
        local_n = base + 1;
        offset = rank * (base + 1);
    } else {
        local_n = base;
        offset = rem * (base + 1) + (rank - rem) * base;
    }

    h = (b - a) / (double)n;
    local_a = a + offset * h;
    local_b = local_a + local_n * h;

    /* Synchronize and start timed portion */
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    /* local computation */
    if (local_n > 0) {
        local_result = local_trap(local_a, local_b, local_n, h);
    } else {
        local_result = 0.0;
    }

    /* reduce results to root */
    MPI_Reduce(&local_result, &total_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* stop timer on root after reduction returns */
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &tend);
        time_sec = (double)(tend.tv_sec - tstart.tv_sec) + (double)(tend.tv_nsec - tstart.tv_nsec) / 1e9;
        printf("n=%ld procs=%d result=%.15f time=%.6f s\n", n, size, total_result, time_sec);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
