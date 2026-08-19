#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  Matrix *kcat = nullptr; // accumulated keys concatenation (in SRAM), shape [r, d]
  Matrix *vcat = nullptr; // accumulated values concatenation (in SRAM), shape [r, d]
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // ---- Build kcat incrementally (keys[0..i]) ----
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    if (kcat == nullptr) {
      kcat = keys[i];
    } else {
      Matrix *newK = matrix_memory_allocator.Allocate("kcat");
      gpu_sim.Concat(kcat, keys[i], newK, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(kcat);
      gpu_sim.ReleaseMatrix(keys[i]);
      kcat = newK;
    }

    // ---- Build vcat incrementally (values[0..i]) ----
    gpu_sim.MoveMatrixToSharedMem(values[i]);
    if (vcat == nullptr) {
      vcat = values[i];
    } else {
      Matrix *newV = matrix_memory_allocator.Allocate("vcat");
      gpu_sim.Concat(vcat, values[i], newV, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(vcat);
      gpu_sim.ReleaseMatrix(values[i]);
      vcat = newV;
    }

    // ---- Move query Q (shape [r, d]) to SRAM ----
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Transpose kcat in place so that (kcat^T) has shape [d, r].
    gpu_sim.Transpose(kcat, kInSharedMemory);

    // S = Q @ Kcat^T  (shape [r, r], r = i+1)
    Matrix *S = matrix_memory_allocator.Allocate("S");
    gpu_sim.MatMul(current_query, kcat, S);

    // Determine the number of rows from the real query matrix (its shape is
    // already valid; S is only populated when Run() executes MatMul).
    size_t r = current_query->GetRowNum();

    // current_query is not needed anymore -> release to save SRAM.
    gpu_sim.ReleaseMatrix(current_query);

    // Transpose kcat back to [r, d] so it can be reused for accumulation next
    // round. (MatMul only reads kcat, so this is safe.)
    gpu_sim.Transpose(kcat, kInSharedMemory);

    // ---- Row-wise softmax of S, compute output row by row. ----
    // out[j] = (exp(S[j]) / sum(exp(S[j]))) @ vcat
    Matrix *out = nullptr;
    for (size_t j = 0; j < r; ++j) {
      Matrix *row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(S, j, row, kInSharedMemory);          // [1, r]
      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row, row_exp);                        // [1, r]
      Matrix *sval = matrix_memory_allocator.Allocate("sum");
      gpu_sim.Sum(row_exp, sval);                          // [1, 1]
      Matrix *w = matrix_memory_allocator.Allocate("w");
      gpu_sim.MatMul(row_exp, vcat, w);                    // [1, d]
      Matrix *outj = matrix_memory_allocator.Allocate("outj");
      gpu_sim.MatDiv(w, sval, outj);                       // [1, d]
      if (out == nullptr) {
        out = outj;
      } else {
        Matrix *newout = matrix_memory_allocator.Allocate("out");
        gpu_sim.Concat(out, outj, newout, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(out);
        gpu_sim.ReleaseMatrix(outj);
        out = newout;
      }
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(sval);
      gpu_sim.ReleaseMatrix(w);
    }
    gpu_sim.ReleaseMatrix(S);

    // Result must be in GPU HBM before commit.
    gpu_sim.MoveMatrixToGpuHbm(out);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*out);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
