# Simplex-Benchmarks
Benchmark code for "Matrix-Free Evaluation Strategies for Continuous and Discontinuous Galerkin Discretizations on Unstructured Tetrahedral Grids"

This repository implements the optimized matrix-free finite-element operator described in
*Still, Fehn, Wall & Kronbichler, "Matrix-Free Evaluation Strategies for Continuous and
Discontinuous Galerkin Discretizations on Unstructured Tetrahedral Grids"* (submitted).

The implementation targets low- to moderate-order polynomial discretizations on tetrahedral meshes and attains high node-level performance by
(i) batching element interpolation into dense matrix–matrix kernels,
(ii) exploiting explicit SIMD data parallelism, and
(iii) applying a hierarchical mesh reordering to improve data locality.
The code is based on the [deal.II](https://www.dealii.org/) library and integrates a hybrid multigrid preconditioner for the Poisson problem used in the paper.

The code has been tested with commit `n5037bf40bd`.

To run an example, go to the corresponding folder, build the benchmark, and run it, e.g.:

```
cmake -D DEAL_II_DIR=/path/to/deal.II .
make -j <Nproc>
mpirun -n <Nproc> ./bench <dim> <degree>
```
Select dim as `3` and degree between `1` and `3`.
To run the hybrid multigrid benchmarks, go to the corresponding folder and build the version of the multigrid preconditioner you want to run.
The two- or three-letter abbreviations specify the transfer operations in the corresponding order.
Further code for the hybrid multigrid can be found [here](https://github.com/dominiktassilostill/exadg/tree/maskedgather2), and optimized hexahedral code can be found [here](https://github.com/kronbichler/multigrid/).
