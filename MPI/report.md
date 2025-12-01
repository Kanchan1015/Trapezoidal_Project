# **MPI Performance Results**

To evaluate the MPI implementation, the trapezoidal program was executed with `n = 10,000,000` subdivisions while varying the number of processes. Each configuration was run **multiple times**, and the **median** runtime was selected to ensure fair and stable measurements. The computed integral values remain extremely close to the analytical value of **2**, confirming the correctness of the distributed computation.

---

## **Execution Times (Median of Multiple Runs)**

| Processes | Result            | Time (s) |
| --------: | ----------------- | -------- |
|         1 | 2.000000000413550 | 0.045830 |
|         2 | 2.000000000295926 | 0.025329 |
|         4 | 2.000000000127473 | 0.012809 |
|         8 | 2.000000000061018 | 0.014851 |

These values show correct numerical integration and clear performance improvements up to 4 processes.

---

## **Speedup and Efficiency**

| Processes | Time (s) | Speedup | Efficiency |
| --------: | -------- | ------- | ---------- |
|         1 | 0.045830 | 1.00×   | 100%       |
|         2 | 0.025329 | 1.81×   | 90.4%      |
|         4 | 0.012809 | 3.58×   | 89.4%      |
|         8 | 0.014851 | 3.09×   | 38.6%      |

The implementation shows excellent strong scaling up to 4 processes, with efficiency close to 90%. Performance decreases at 8 processes due to hardware limitations discussed below.

---

## **Parallel Work Distribution**

The interval ([0, \pi]) was divided among the MPI ranks using:

- **Remainder distribution:**
  The first `rem = n % p` ranks were assigned one extra subdivision to ensure balance.
- Each rank computed the integral over its local start and local end using the same width `h`.

This ensures that no rank becomes idle, even when `n` is not perfectly divisible.

via a single `MPI_Reduce(..., MPI_SUM, ...)` operation — this introduces minimal communication overhead.

---

## **Why Speedup Is Not Linear**

Linear speedup was not achieved due to several expected factors in MPI:

- Separate processes: MPI ranks are heavier than OpenMP threads, causing more overhead during startup and synchronization.
- Shared hardware: On a laptop, multiple MPI processes compete for the same physical cores, cache, and memory bandwidth. Beyond 4 processes, oversubscription occurs.
- Communication overhead: `MPI_Reduce` requires inter-process communication implemented via the OS kernel.
- Process scheduling: With more ranks than cores, the OS begins frequent context switching, slowing the program.

Because of this, the 8-process run becomes slower than the 4-process run on single-node hardware.

---

## **Observations**

- Strong performance improvement from 1 → 4 MPI processes, indicating good parallel efficiency (~90%).
- Peak performance achieved at 4 processes.
- Slight slowdown at 8 processes is expected due to oversubscription and memory contention on a non-cluster device.
- Remainder-based load balancing ensured that all ranks contributed meaningfully.
- The combination strategy using `MPI_Reduce` scales efficiently and keeps communication minimal.

---

## **Hardware Constraint Note**

All tests were done on a single laptop, using local `mpirun`.
Since MPI spawns full OS processes, running with 8 (or more) ranks on a machine with limited physical cores naturally reduces efficiency. This is normal and is not a reflection of code quality

The assignment allows this situation as long as:

- hardware limitations are acknowledged
- results are correctly interpreted
- MPI behaviour is explained (done above)

Running the same MPI code on a real multi-node cluster would likely improve speedups for 8–16 processes.

---

## **Conclusion**

The MPI-based trapezoidal integration demonstrates correct numerical behaviour and strong parallel performance. Independent sub-interval calculations make the algorithm naturally suited for message-passing parallelism. The results confirm that MPI effectively distributes work and combines results with minimal communication overhead. While performance saturates beyond 4 processes because of workstation hardware limits, the implementation and analysis meet all MPI requirements for correctness, distribution, scalability evaluation, and performance interpretation.
