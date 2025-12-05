import matplotlib.pyplot as plt
import os

# --- OpenMP measured data ---
threads = [1, 2, 3, 4, 8]
times = [0.048822, 0.036610, 0.026622, 0.021934, 0.015739]

# --- Compute speedup ---
t1 = times[0]
speedup = [t1 / t for t in times]

# --- Make output directory ---
out_dir = "OpenMP/screenshots/graphs"
os.makedirs(out_dir, exist_ok=True)

# --- Time vs Threads ---
plt.figure()
plt.plot(threads, times, marker='o')
plt.xlabel('Threads')
plt.ylabel('Time (s)')
plt.title('OpenMP: Time vs Threads')
plt.grid(True)
plt.savefig(f"{out_dir}/OpenMP_time_vs_threads.png", bbox_inches='tight')

# --- Speedup vs Threads ---
plt.figure()
plt.plot(threads, speedup, marker='o')
plt.xlabel('Threads')
plt.ylabel('Speedup')
plt.title('OpenMP: Speedup vs Threads')
plt.grid(True)
plt.savefig(f"{out_dir}/OpenMP_speedup_vs_threads.png", bbox_inches='tight')

print("Graphs created in OpenMP/screenshots/graphs/")
