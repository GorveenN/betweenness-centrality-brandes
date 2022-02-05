#ifndef STRIDE_CSR_GRAPH_CUH
#define STRIDE_CSR_GRAPH_CUH

#include <utility>
#include <vector>

struct GPUStrideCSRGraph {
    int* offset;
    int* vmap;
    int* nvir;
    int* ptrs;
    int* adjs;
    size_t num_vertices;
    size_t num_connected_vertices;
    size_t num_virtual_vertices;
    ~GPUStrideCSRGraph();
};

struct CPUStrideCSRGraph {
    std::vector<int> offset;
    std::vector<int> vmap;
    std::vector<int> nvir;
    std::vector<int> ptrs;
    std::vector<int> adjs;
    size_t num_vertices;
    size_t num_connected_vertices;
    size_t num_virtual_vertices;

    CPUStrideCSRGraph(const std::vector<std::pair<int, int>>, int);

    GPUStrideCSRGraph to_gpu() const;

    void print_raw() const;
};

#endif  // STRIDE_CSR_GRAPH_CUH
