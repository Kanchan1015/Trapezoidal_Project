1. Compile

---

mpicc -O2 -lm -o trapezoidal_mpi trapezoidal_mpi.c

---

2. Run

---

mpirun -np <processes> ./trapezoidal_mpi <n>

Examples:
mpirun -np 1 ./trapezoidal_mpi 10000000
mpirun -np 4 ./trapezoidal_mpi 10000000

On some systems use "mpiexec" instead of "mpirun".

---

3. Choosing number of processes

---

The number of processes is controlled only by mpirun/mpiexec.
The program automatically divides the interval among all ranks,
handling cases where n is not divisible by the number of processes.

---

4. Timing note

---

For fair comparisons:

- keep the same n for all runs,
- run each configuration multiple times and average/choose median.

---

5. Cluster note

---

If running on a cluster, use your scheduler or:
mpirun -np <p> --hostfile myhosts ./trapezoidal_mpi <n>

If running locally, normal single-machine mpirun is sufficient.
