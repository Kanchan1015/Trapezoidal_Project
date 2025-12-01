## OpenMP Trapezoidal Rule

Source file:
trapezoidal_omp.c

Compile (Linux/macOS with OpenMP):
gcc-15 -fopenmp -O2 trapezoidal_omp.c -o trapezoidal_omp -lm

Run:
./trapezoidal_omp <n> <threads>

Examples:
./trapezoidal_omp 10000000 1
./trapezoidal_omp 10000000 4
./trapezoidal_omp 10000000 8

Notes:

- The -O2 flag gives better optimized performance.
- The -fopenmp flag enables OpenMP pragmas.
- Thread count can be set either by argv[2] or the OMP_NUM_THREADS environment variable.
- Output format is:
  n=<n> threads=<t> result=<value> time=<seconds> s
