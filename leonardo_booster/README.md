# Leonardo Booster — Node Architecture

Architecture notes for the Leonardo Booster partition at Cineca. The `arch/` folder has the node topology captured from `lstopo`.

## Node layout

2 sockets (Intel Xeon Platinum 8358), 8 NUMA nodes total — 4 per socket, 14 cores each, 112 cores per node. Each socket has a 105 MB shared L3. Memory is 503 GB split across the 8 NUMA nodes (~63 GB each).

Interconnect is InfiniBand (mlx5_0). GPUs (A100) attach via PCIe on each socket.

## Why this matters

Getting good performance on a multi-socket, multi-NUMA machine means being deliberate about where your processes and threads land:

- **NUMA locality** — an MPI rank reading memory allocated on a remote NUMA node takes a penalty. Pinning ranks to cores within the same NUMA domain as their data keeps bandwidth local.
- **MPI rank placement** — with 4 NUMA nodes per socket and typically 4 ranks per node in our jobs (`--ntasks-per-socket=4`), each rank maps to one NUMA domain. This avoids cross-NUMA memory traffic during halo exchanges.
- **OpenMP thread binding** — `OMP_PROC_BIND=close` + `OMP_PLACES=cores` keeps threads in the same NUMA node as the rank that spawned them. Spreading threads across NUMA nodes kills bandwidth for stencil-heavy workloads.
- **GPU affinity** — on Leonardo each socket has 2 A100s attached. Binding a rank to the wrong socket means its GPU transfers cross the PCIe root complex, adding latency. The SLURM scripts in this repo set `--ntasks-per-socket` and `--cpus-per-task` to match this topology.

The SLURM flags used across the jobs in this repo (`--ntasks-per-socket=4`, `--cpus-per-task=14`, `--hint=nomultithread`) were chosen with this layout in mind.
