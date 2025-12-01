Serial Implementation (Trapezoidal Rule)
The serial implementation computes the definite integral using the Trapezoidal Rule by dividing the interval [a,b] into n equal subintervals and summing the areas of trapezoids. The code is implemented in C and measures wall-clock execution time using clock_gettime(CLOCK_MONOTONIC). The function used for verification is f(x)=sin(x) with interval [0, π], for which the exact integral is 2.0. The serial program was executed with multiple subdivision counts (n = 1e6, 5e6, 1e7, 2e7) to generate baseline execution times for later comparison with OpenMP, MPI, and CUDA implementations.

AI Assistance
The serial code structure, documentation guidance, and explanation sections were prepared with assistance from ChatGPT (OpenAI). The exact prompts used are listed in the file prompts/serial.md included in the submission package.
