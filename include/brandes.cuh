#ifndef STRIDE_CSR_BRANDES_CUH
#define STRIDE_CSR_BRANDES_CUH

#include <graph.cuh>

double* run_stride_csr(const CPUStrideCSRGraph& graph, float& time, bool verbose = false);

#endif  // STRIDE_CSR_BRANDES_CUH
