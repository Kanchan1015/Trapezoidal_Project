import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Your measured times (replace here if needed) ---
processes = [1, 2, 4, 8]
times = [0.045830, 0.025329, 0.012809, 0.014851]

# --- Compute speedup ---
t1 = times[0]
speedup = [t1 / t for t in times]

# --- Make output directory ---
out_dir = "MPI/screenshots/graphs"
os.makedirs(out_dir, exist_ok=True)

# --- Time vs Processes ---
plt.figure()
plt.plot(processes, times, marker='o')
plt.xlabel('Processes')
plt.ylabel('Time (s)')
plt.title('MPI: Time vs Processes')
plt.grid(True)
plt.savefig(f"{out_dir}/MPI_time_vs_procs.png", bbox_inches='tight')

# --- Speedup vs Processes ---
plt.figure()
plt.plot(processes, speedup, marker='o')
plt.xlabel('Processes')
plt.ylabel('Speedup')
plt.title('MPI: Speedup vs Processes')
plt.grid(True)
plt.savefig(f"{out_dir}/MPI_speedup_vs_procs.png", bbox_inches='tight')

print("Graphs created in Screenshots/MPI/Graphs/")
