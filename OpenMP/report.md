# **OpenMP Performance Results **

To evaluate the OpenMP implementation, the program was executed with `n = 10,000,000` subdivisions while varying the number of threads. The table below summarizes the computed integral and execution time for each configuration. All results are very close to the expected value of 2, confirming correctness.

### **Execution Times**

| Threads | Result            | Time (s) |
| ------: | ----------------- | -------- |
|       1 | 1.999999999999809 | 0.048822 |
|       2 | 1.999999999999961 | 0.036610 |
|       3 | 2.000000000000148 | 0.026622 |
|       4 | 2.000000000000020 | 0.021934 |
|       8 | 1.999999999999960 | 0.015739 |

### **Speedup Calculation**

Speedup was calculated as:
Speedup = T₁ / Tₙ

| Threads | Time (s) | Speedup |
| ------: | -------- | ------- |
|       1 | 0.048822 | 1.00×   |
|       2 | 0.036610 | 1.33×   |
|       3 | 0.026622 | 1.83×   |
|       4 | 0.021934 | 2.22×   |
|       8 | 0.015739 | 3.10×   |

The program shows consistent speedup as threads increase, with diminishing returns after 4–8 threads. This is expected because the cost per iteration is uniform and the loop is evenly split across threads using `schedule(static)`.

### **Why Speedup Is Not Linear**

Speedup does not scale perfectly due to:

- Thread scheduling overhead when creating and managing multiple threads
- Memory bandwidth limits, especially as more threads access the same memory region
- Reduction overhead, since partial sums must be combined at the end
- CPU core limits, since the system used for testing supports a maximum of _X_ hardware threads (replace X with your Mac’s number)

These factors introduce small delays that prevent ideal 8× speedup with 8 threads.

### **Observations**

- The runtime decreases rapidly from 1 → 3 threads, showing strong scaling.
- The best performance was achieved at 8 threads, giving a 3.1× speedup.
- Static scheduling worked well due to the uniform cost of evaluating `sin(x)`.
- The reduction clause ensures safe accumulation without race conditions, with only a small synchronization cost at the end.

### **Note on Hardware Constraints**

The Apple M-series processor used for testing supports up to 8 logical threads
Because of this, running with 16 threads was not possible on this device. The assignment allows this limitation as long as it is acknowledged.

### **Conclusion**

The trapezoidal rule is a parallel algorithm because each iteration is independent. This makes OpenMP a perfect fit, and the implementation achieves significant speedup with minimal parallel overhead. The results clearly demonstrate the benefits of thread-level parallelism for numerical integration workloads.
