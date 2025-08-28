# simplex-benchmarks
Benchmark code for "Matrix-Free Evaluation Strategies for Continuous and Discontinuous Galerkin Discretizations on Unstructured Tetrahedral Grids"
This repository implements the optimized matrix-free finite-element operator described in
*Still, Fehn, Wall & Kronbichler, "Matrix-Free Evaluation Strategies for Continuous and
Discontinuous Galerkin Discretizations on Unstructured Tetrahedral Grids"* (submitted).
The implementation targets low- to moderate-order polynomial discretizations on tetrahedral
meshes and attains high node-level performance by (i) batching element interpolation into
dense matrix–matrix kernels, (ii) exploiting explicit SIMD data-parallelism, and
(iii) applying a hierarchical mesh-reordering to improve data locality. The code is based
on the [deal.II](https://www.dealii.org/) library and integrates a hybrid multigrid
preconditioner for the Poisson problem used in the paper.

The code is build on the deal.ii libary (https://github.com/dealii/dealii) and runs with version 5037bf40bd.

To run an example, go to the corresponding folder, build the benchmark and run it, e.g. by

```
cmake -D DEAL_II_DIR=/path/to/deal.II .
make -j <Nproc>
mpirun -n <Nproc> ./bench <dim> <degree>
```

