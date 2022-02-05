#include <cstddef>
#include <iostream>
#include <map>
#include <unordered_set>
#include "graph.cuh"

CPUStrideCSRGraph::CPUStrideCSRGraph(const std::vector<std::pair<int, int>> edges, int mdeg) {
    std::map<int, std::unordered_set<int>> graph;
    int num_vertices = -1;
    for (auto const& edge : edges) {
        num_vertices = std::max(num_vertices, std::max(edge.first, edge.second));
        graph[edge.first].emplace(edge.second);
        graph[edge.second].emplace(edge.first);
    }
    num_vertices++;

    int num_edges_processed = 0;
    for (int vertex = 0; vertex < num_vertices; vertex++) {
        std::unordered_set<int> edges;
        auto iter = graph.find(vertex);
        if (iter != graph.end()) {
            edges = iter->second;
        }

        int num_nvir = std::ceil(float(edges.size()) / float(mdeg));
        this->nvir.push_back(num_nvir);
        for (int i = 0; i < num_nvir; i++) {
            this->offset.push_back(i);
            this->vmap.push_back(vertex);
        }

        this->ptrs.push_back(num_edges_processed);
        for (const int& edge : edges) {
            num_edges_processed++;
            this->adjs.push_back(edge);
        }
    }
    this->ptrs.push_back(num_edges_processed);

    this->num_connected_vertices = edges.size();
    this->num_vertices = num_vertices;
    this->num_virtual_vertices = this->offset.size();
}

GPUStrideCSRGraph CPUStrideCSRGraph::to_gpu() const {
    GPUStrideCSRGraph gpu_graph;
    cudaMalloc(&gpu_graph.offset, this->offset.size() * sizeof(int));
    cudaMalloc(&gpu_graph.vmap, this->vmap.size() * sizeof(int));
    cudaMalloc(&gpu_graph.nvir, this->nvir.size() * sizeof(int));
    cudaMalloc(&gpu_graph.ptrs, this->ptrs.size() * sizeof(int));
    cudaMalloc(&gpu_graph.adjs, this->adjs.size() * sizeof(int));

    cudaMemcpy(gpu_graph.offset, this->offset.data(), this->offset.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_graph.vmap, this->vmap.data(), this->vmap.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_graph.nvir, this->nvir.data(), this->nvir.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_graph.ptrs, this->ptrs.data(), this->ptrs.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_graph.adjs, this->adjs.data(), this->adjs.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    gpu_graph.num_vertices = this->num_vertices;
    gpu_graph.num_connected_vertices = this->num_connected_vertices;
    gpu_graph.num_virtual_vertices = this->num_virtual_vertices;

    return gpu_graph;
}

void print_vec(const std::vector<int>& vec) {
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << vec[i];
    }
    std::cout << "]" << std::endl;
}

void CPUStrideCSRGraph::print_raw() const {
    std::cout << "offset: ";
    print_vec(this->offset);

    std::cout << "vmap: ";
    print_vec(this->vmap);

    std::cout << "nvir: ";
    print_vec(this->nvir);

    std::cout << "ptrs: ";
    print_vec(this->ptrs);

    std::cout << "adjs: ";
    print_vec(this->adjs);
}

GPUStrideCSRGraph::~GPUStrideCSRGraph() {
    cudaFree(this->offset);
    cudaFree(this->vmap);
    cudaFree(this->nvir);
    cudaFree(this->ptrs);
    cudaFree(this->adjs);
}
