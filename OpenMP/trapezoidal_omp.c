#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

/* f(x) = sin(x) */
static inline double f(double x) { return sin(x); }

/* simple wall-clock timer */
double elapsed_seconds(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1e9;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s n [num_threads]\n", argv[0]);
        return 1;
    }

    long n = atol(argv[1]);               /* number of subdivisions */
    if (n <= 0) {
        fprintf(stderr, "n must be > 0\n");
        return 1;
    }

    int requested_threads = 0;
    if (argc >= 3) requested_threads = atoi(argv[2]); /* thread count */
    if (requested_threads > 0) omp_set_num_threads(requested_threads);

    int threads_report = (requested_threads > 0) ? requested_threads : omp_get_max_threads();

    double a = 0.0, b = PI;
    double h = (b - a) / (double)n;
    double sum = 0.0;

    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    /* endpoints contribution (handled once) */
    sum = 0.5 * (f(a) + f(b));

    /* parallel interior sum with safe reduction */
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 1; i <= n - 1; ++i) {
        double x = a + i * h;
        sum += f(x);
    }

    double result = sum * h;

    clock_gettime(CLOCK_MONOTONIC, &tend);
    double time_s = elapsed_seconds(&tstart, &tend);

    /* required output format */
    printf("n=%ld threads=%d result=%.15f time=%.6f s\n", n, threads_report, result, time_s);
    return 0;
}
