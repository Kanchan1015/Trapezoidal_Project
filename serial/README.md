Serial implementation - Trapezoidal Rule

Files:

- trapezoidal.c : Serial C source code
- serial_run_results.csv : Collected timing results
- Screenshots/Serial/ : Compile and run screenshots

Compile (Linux/macOS):
gcc trapezoidal.c -o trap -lm -O2

Run:
./trap # default n
./trap 10000000 # run with 10,000,000 subdivisions

Notes:

- The program prints: n=... result=... time=... s
- Use sufficiently large n (1e6 to 1e8) to get measurable timing.
- Append recorded runs to serial_run_results.csv for later plotting/comparison.
