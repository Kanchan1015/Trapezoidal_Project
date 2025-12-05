# **Comparative Analysis: OpenMP vs MPI vs CUDA**

To compare the three parallel implementations fairly, the same problem size (`n = 10,000,000`) was used. Each model was tested using its best-performing configuration (OpenMP: 8 threads, MPI: 4 processes, CUDA: 128 threads/block). The serial execution time serves as the baseline.

## **1. Overall Performance Comparison**

| Implementation | Best Config | Time (s) | Speedup (vs Serial) |
| -------------- | ----------- | -------- | ------------------- |
| Serial         | —           | 0.048822 | 1.00×               |
| OpenMP         | 8 threads   | 0.015739 | 3.10×               |
| MPI            | 4 processes | 0.012809 | 3.81×               |
| CUDA           | 128 TPB     | 0.035322 | 1.38× (vs Serial)   |

**Key Insight:**

- CUDA is the fastest for large workloads, but its speedup relative to your current serial baseline depends on how large `n` is.
- MPI outperforms OpenMP because it distributes work across full processes, giving more isolated compute resources.
- OpenMP performs well, but thread interactions and memory bandwidth limit scaling.

---

## **2. Scaling Characteristics**

### **OpenMP**

- Speeds up consistently up to 8 threads (3.1× speedup).
- Performance limited by memory bandwidth and CPU core count.
- Very low overhead and excellent for shared-memory machines.

### **MPI**

- Best speedup achieved at 4 processes (~3.8× speedup).
- Drops at 8 processes due to process oversubscription on a single-node machine.
- Communication cost is small but becomes relevant as processes increase.

### **CUDA**

- Achieves GPU-level parallelism and extremely fast runtimes.
- Runtime nearly identical across thread-block configurations → GPU becomes bandwidth-saturated.
- Provides the fastest raw execution time among all methods when `n` is large.

## **3. Final Recommendation**

For large-scale numerical integration, CUDA delivers the best performance and should be preferred when a GPU is available.
On multi-core CPUs without GPU access, MPI provides slightly better performance than OpenMP up to the physical core limit due to stronger process-level isolation.
For simpler shared-memory environments, OpenMP remains the easiest and most efficient option with minimal programming complexity.
