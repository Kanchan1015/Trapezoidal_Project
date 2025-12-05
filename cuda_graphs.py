import matplotlib.pyplot as plt
import os

# --- CUDA measured data (from your CSV results) ---
threads = [64, 128, 256, 512]
times = [0.035320, 0.035322, 0.035305, 0.035316]

# n vs time (we use 1M, 5M, 10M results)
n_values = [1_000_000, 5_000_000, 10_000_000]
n_times = [0.003732, 0.017799, 0.035320]   # time at 64 threads for 10M

# --- Compute speedup for threads-per-block graph ---
baseline = times[0]   # time at 64 threads
speedup = [baseline / t for t in times]

# --- Output directory ---
out_dir = "CUDA/screenshots/graphs"
os.makedirs(out_dir, exist_ok=True)

# ==============================
# 1) Time vs ThreadsPerBlock
# ==============================
plt.figure()
plt.plot(threads, times, marker='o')
plt.xlabel('Threads per Block')
plt.ylabel('Time (s)')
plt.title('CUDA: Time vs ThreadsPerBlock')
plt.grid(True)
plt.savefig(f"{out_dir}/CUDA_time_vs_threads.png", bbox_inches='tight')

# ==============================
# 2) Speedup vs ThreadsPerBlock
# ==============================
plt.figure()
plt.plot(threads, speedup, marker='o')
plt.xlabel('Threads per Block')
plt.ylabel('Speedup')
plt.title('CUDA: Speedup vs ThreadsPerBlock')
plt.grid(True)
plt.savefig(f"{out_dir}/CUDA_speedup_vs_threads.png", bbox_inches='tight')

# ==============================
# 3) Time vs Problem Size (n)
# ==============================
plt.figure()
plt.plot(n_values, n_times, marker='o')
plt.xlabel('n (Number of Subdivisions)')
plt.ylabel('Time (s)')
plt.title('CUDA: Time vs Problem Size (n)')
plt.grid(True)
plt.savefig(f"{out_dir}/CUDA_time_vs_n.png", bbox_inches='tight')

print("CUDA graphs created in CUDA/screenshots/graphs/")
