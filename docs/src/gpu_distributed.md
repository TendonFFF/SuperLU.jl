# GPU and Distributed Computing

## Overview

SuperLU.jl currently provides a CPU-based interface to the serial SuperLU library. For GPU acceleration and distributed computing capabilities, the SuperLU family offers additional libraries.

## SuperLU Library Variants

The SuperLU project consists of three main components:

| Library | Purpose | Julia Package |
|---------|---------|---------------|
| **SuperLU** | Sequential (single-threaded) sparse LU | [SuperLU.jl](https://github.com/TendonFFF/SuperLU.jl) (this package) |
| **SuperLU_MT** | Multi-threaded shared memory | SuperLU_MT_jll (binaries only) |
| **SuperLU_DIST** | Distributed memory + GPU | [SuperLUDIST.jl](https://github.com/JuliaSparse/SuperLUDIST.jl) |

## GPU Acceleration via SuperLU_DIST

GPU acceleration in SuperLU is available through the **SuperLU_DIST** library, which is designed for distributed memory systems but can also be used effectively on a single workstation with multiple GPUs.

### How SuperLU_DIST GPU Support Works

SuperLU_DIST supports GPU acceleration for:

- **Numerical Factorization (GSTRF)**: The compute-intensive factorization phase can utilize GPU cores
- **Triangular Solve (GSTRS)**: Forward and backward substitution can be GPU-accelerated
- **Batched Operations**: Multiple right-hand sides can be solved efficiently on GPU

### Hardware Requirements

- NVIDIA GPU with CUDA support
- Sufficient GPU memory for matrix factors
- CUDA toolkit and cuBLAS library

### Julia Integration: SuperLUDIST.jl

For GPU and distributed computing with SuperLU, use [SuperLUDIST.jl](https://github.com/JuliaSparse/SuperLUDIST.jl):

```julia
using Pkg
Pkg.add("SuperLUDIST")
```

**Note:** As of the current version, CUDA support in SuperLUDIST.jl is disabled. Check the package documentation for updates on GPU support.

### Example: Single-Node Multi-GPU Setup (e.g., 4U Server with 2 L40S GPUs)

For a single workstation with multiple NVIDIA GPUs (such as 2 L40S GPUs), SuperLU_DIST can be configured to:

1. Run 2-4 MPI processes on the single node
2. Assign one GPU to each MPI process
3. Distribute the matrix across processes
4. Utilize GPU acceleration for factorization and solve

```julia
# Example setup (requires MPI.jl and SuperLUDIST.jl)
using MPI
using SuperLUDIST

MPI.Init()

# Configure process grid for 2 GPUs
nprow, npcol = 1, 2  # 1x2 grid for 2 processes
grid = Grid{Int64}(nprow, npcol, MPI.COMM_WORLD)

# ... matrix setup and solve
```

### Performance Considerations

GPU acceleration is most beneficial for:

- **Large sparse systems** (N > 50,000, with noticeable benefits starting around N > 10,000)
- **Matrices with dense supernodes** (common in FEM/CFD applications)
- **Multiple solves with the same factorization**

For smaller systems, the overhead of GPU data transfer may outweigh the computational benefits.

### Choosing Between SuperLU.jl and SuperLUDIST.jl

| Use Case | Recommended Package |
|----------|---------------------|
| Small to medium systems (N < 10,000) | SuperLU.jl |
| Medium systems (10,000 < N < 50,000) | Either (benchmark to decide) |
| Large systems (N > 50,000) | SuperLUDIST.jl |
| Single-threaded CPU only | SuperLU.jl |
| Multi-GPU workstation | SuperLUDIST.jl |
| HPC cluster / distributed memory | SuperLUDIST.jl |
| Need maximum compatibility | SuperLU.jl |

## NVIDIA L40S GPU Considerations

The NVIDIA L40S is particularly well-suited for SuperLU_DIST workloads:

- **48 GB GDDR6 memory** per GPU: Large enough for substantial sparse matrices
- **91.6 TFLOPS FP32**: High compute throughput for dense supernodal operations
- **PCIe Gen4 x16**: Fast CPU-GPU and GPU-GPU communication

For a 4U server with 2 L40S GPUs:
- Can handle sparse systems with millions of unknowns
- Optimal performance when each GPU processes a substantial portion of the matrix
- Consider using 2-4 MPI ranks (1-2 per GPU)

## Future Plans

The SuperLU.jl package may add:
- Unified interface for selecting CPU vs GPU backends
- Automatic backend selection based on problem size
- Simplified multi-GPU support for single-node deployments

For updates, watch this repository and [SuperLUDIST.jl](https://github.com/JuliaSparse/SuperLUDIST.jl).

## References

- [SuperLU Homepage](https://portal.nersc.gov/project/sparse/superlu/)
- [SuperLU_DIST Repository](https://github.com/xiaoyeli/superlu_dist)
- [SuperLUDIST.jl Documentation](https://aa25desh.github.io/SuperLUDIST.jl/stable/)
- [MPI.jl Documentation](https://juliaparallel.org/MPI.jl/stable/)
