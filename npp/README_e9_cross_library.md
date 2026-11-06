# E9 cross-library kernel (cvGPUSpeedup + FastNPP + a user operation in one `fk::executeOperations`)

Program `e9_cross_library.cu` (target `e9_cross_library_cu`); design, predictions and what the demonstration does and does not show are in its header comment. Four kernels: M1, M2 (mixed), R1 (all cvGS), R2 (all FastNPP), the same user gamma in all.

- **Listing:** lines 226-233 of `e9_cross_library.cu` (the call in the lambda `listing`), a comment on each line names the origin of the operation.
- **Prediction (2) was false** (dated note in the header): M and R differ by 8 to 16 SASS lines (M1 34 registers / 552 lines, R1 32 / 544, M2 33 / 544, R2 32 / 560; stack, shared, local 0) because the swap and the constant arithmetic are different FKL types in the two wrappers. The SASS comparison is a size comparison, not an identity test, and the time comparison M against R contains that difference as well.
- **Friction in the first draft** (an ambiguous `Gamma::build`) was a mistake of the program, not a finding about the libraries.
- Results (bit-exact check, kernel counts, times) come from the single window after FastNPP #45 is merged and the pin is moved; none exist yet.
