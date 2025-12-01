// trapezoidal.c
// Assistance: ChatGPT (OpenAI)

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>


double f(double x) { return sin(x); } 

int main(int argc, char **argv) {
    double a = 0.0, b = M_PI;
    long long n = 1000000; 
    if (argc > 1) n = atoll(argv[1]);

    double h = (b - a) / (double)n;
    double sum = 0.0;

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    for (long long i = 1; i < n; i++) sum += f(a + i*h);

    double result = (h/2.0) * (f(a) + f(b) + 2.0*sum);

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) * 1e-9;

    printf("n=%lld  result=%.10f  time=%.6f s\n", n, result, elapsed);
    return 0;
}
