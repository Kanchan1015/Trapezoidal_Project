CUDA Implementation - Trapezoidal Rule

Files:

- trapezoidal_cuda.cu : CUDA source code
- trap_cuda : compiled binary
- cuda_raw_output.txt : raw run outputs
- cuda_run_results.csv : cleaned CSV data for graphs

How to compile (Colab):
nvcc trapezoidal_cuda.cu -O2 -arch=sm_75 -o trap_cuda

How to run:
./trap_cuda <n> <threadsPerBlock>
Example:
./trap_cuda 10000000 128

Notes:

- Uses Tesla T4 GPU provided by Colab.
- Uses atomicAdd(double) – GPU must support compute capability >= 6.0.
- Kernel uses striding pattern to cover all points.
- Timing uses CUDA events.
- CSV file is used for plotting performance graphs.
