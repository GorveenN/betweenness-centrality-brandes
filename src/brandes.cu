//#include "../utils.cuh"
#include <stdio.h>
#include <brandes.cuh>
#include <graph.cuh>
#include <iostream>
#include <utils.cuh>

__global__ void brandes_forward_kernel(const int* __restrict__ g_offset,
                                       const int* __restrict__ g_vmap,
                                       const int* __restrict__ g_nvir,
                                       const int* __restrict__ g_ptrs,
                                       const int* __restrict__ g_adjs,
                                       int g_nvert,
                                       int* dist,
                                       int* sigma,
                                       int* cont,
                                       int level) {
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int virt_vertex = global_thread_idx; virt_vertex < g_nvert; virt_vertex += num_threads) {
        int source_vertex = g_vmap[virt_vertex];
        int nth_virt = g_offset[virt_vertex];
        int stride = g_nvir[source_vertex];

        if (dist[source_vertex] != level) {
            continue;
        }

        int edges_idx_min = g_ptrs[source_vertex];
        int edges_idx_max = g_ptrs[source_vertex + 1];

        for (int neighbour_idx = edges_idx_min + nth_virt; neighbour_idx < edges_idx_max;
             neighbour_idx += stride) {
            int neighbour = g_adjs[neighbour_idx];

            if (dist[neighbour] == -1) {
                dist[neighbour] = level + 1;

                *cont = 1;
            }

            if (dist[neighbour] == level + 1) {
                atomicAdd(sigma + neighbour, sigma[source_vertex]);
            }
        }
    }
}

__global__ void brandes_backward_kernel(const int* __restrict__ g_offset,
                                        const int* __restrict__ g_vmap,
                                        const int* __restrict__ g_nvir,
                                        const int* __restrict__ g_ptrs,
                                        const int* __restrict__ g_adjs,
                                        int g_nvert,
                                        int* dist,
                                        double* delta,
                                        int level) {
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int virt_vertex = global_thread_idx; virt_vertex < g_nvert; virt_vertex += num_threads) {
        int source_vertex = g_vmap[virt_vertex];
        int nth_virt = g_offset[virt_vertex];
        int stride = g_nvir[source_vertex];
        if (dist[source_vertex] == level) {
            int edges_idx_min = g_ptrs[source_vertex];
            int edges_idx_max = g_ptrs[source_vertex + 1];

            double sum = 0;

            for (int neigh_idx = edges_idx_min + nth_virt; neigh_idx < edges_idx_max;
                 neigh_idx += stride) {
                int neigh = g_adjs[neigh_idx];

                // source is predecessor of neighbour
                if (dist[neigh] == level + 1) {
                    sum += delta[neigh];
                }
            }

            atomicAdd(delta + source_vertex, sum);
        }
    }
}

__global__ void brandes_vertex_sum_kernel(const int source,
                                          const int num_vertices,
                                          const int* __restrict__ distance,
                                          const double* __restrict__ deltas,
                                          const int* __restrict__ sigmas,
                                          double* betweenness) {
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int source_vertex = global_thread_idx; source_vertex < num_vertices;
         source_vertex += num_threads) {
        if (source_vertex == source) {
            continue;
        }

        if (distance[source_vertex] == -1) {
            continue;
        }

        betweenness[source_vertex] += deltas[source_vertex] * (double)sigmas[source_vertex] - 1.;
    }
}

__global__ void brandes_init_deltas(const int num_vertices,
                                    double* delta,
                                    const int* __restrict__ sigmas) {
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int vertex = global_thread_idx; vertex < num_vertices; vertex += num_threads) {
        int sigma = sigmas[vertex];
        delta[vertex] = sigma != 0 ? 1. / sigma : 0;
    }
}

int brandes_forward(const GPUStrideCSRGraph& graph,
                    int source,
                    int* distance,
                    int* sigma,
                    int* d_cont) {
    int level = 0;
    int h_cont = 1;

    // fill with -1, 0 for source
    cudaMemcpyTyped(distance, -1, graph.num_vertices);
    cudaMemcpyTyped(distance + source, 0, 1);

    // fill with 1, 0 for source
    HANDLE_ERROR(cudaMemset(sigma, 0, graph.num_vertices * sizeof(int)));
    cudaMemcpyTyped(sigma + source, 1, 1);

    while (h_cont) {
        brandes_forward_kernel<<<int(graph.num_virtual_vertices / NUM_THREADS) + 1, NUM_THREADS>>>(
            graph.offset, graph.vmap, graph.nvir, graph.ptrs, graph.adjs,
            graph.num_virtual_vertices, distance, sigma, d_cont, level);

        HANDLE_ERROR(cudaMemcpy(&h_cont, d_cont, sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemset(d_cont, 0, sizeof(int)));
        level++;
    }

    return level;
}

void brandes_backward(const GPUStrideCSRGraph& graph,
                      int level,
                      int* d_distance,
                      int* d_sigma,
                      double* d_delta) {
    brandes_init_deltas<<<int(graph.num_virtual_vertices / NUM_THREADS) + 1, NUM_THREADS>>>(
        graph.num_vertices, d_delta, d_sigma);

    while (level > 1) {
        level--;
        brandes_backward_kernel<<<int(graph.num_virtual_vertices / NUM_THREADS) + 1, NUM_THREADS>>>(
            graph.offset, graph.vmap, graph.nvir, graph.ptrs, graph.adjs,
            graph.num_virtual_vertices, d_distance, d_delta, level);
    }
}

double* run_stride_csr(const CPUStrideCSRGraph& graph, float& time, bool verbose) {
    double* h_betweenness = new double[graph.num_vertices];

    float elapsed_ker_mem = 0, elapsed_ker = 0;
    CUDATimer timer_ker_mem, timer_ker;

    timer_ker_mem.start();
    double* d_betweenness;
    GPUStrideCSRGraph d_graph = graph.to_gpu();

    HANDLE_ERROR(cudaMalloc(&d_betweenness, graph.num_vertices * sizeof(double)));
    cudaMemcpyTyped(d_betweenness, double(0.), graph.num_vertices);
    int *d_distance, *d_sigmas, *d_cont;
    double* d_delta;

    HANDLE_ERROR(cudaMalloc(&d_distance, graph.num_vertices * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_sigmas, graph.num_vertices * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_delta, graph.num_vertices * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_cont, sizeof(int)));

    for (int source = 0; source < graph.num_vertices; source++) {
        if (source % 1000 == 0 && verbose) {
            std::cout << "running brandes from " << source << " vertex" << std::endl;
        }

        int level = 0;
        int h_cont = 1;

        // fill with -1, 0 for source
        cudaMemcpyTyped(d_distance, -1, graph.num_vertices);
        cudaMemcpyTyped(d_distance + source, 0, 1);

        // fill with 1, 0 for source
        HANDLE_ERROR(cudaMemset(d_sigmas, 0, graph.num_vertices * sizeof(int)));
        cudaMemcpyTyped(d_sigmas + source, 1, 1);

        while (h_cont) {
            timer_ker.start();
            brandes_forward_kernel<<<int(graph.num_virtual_vertices / NUM_THREADS) + 1,
                                     NUM_THREADS>>>(
                d_graph.offset, d_graph.vmap, d_graph.nvir, d_graph.ptrs, d_graph.adjs,
                d_graph.num_virtual_vertices, d_distance, d_sigmas, d_cont, level);
            timer_ker.stop();
            elapsed_ker += timer_ker.elapsed();

            HANDLE_ERROR(cudaMemcpy(&h_cont, d_cont, sizeof(int), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemset(d_cont, 0, sizeof(int)));
            level++;
        }
        // there is no neighbours to source, skip
        if (graph.ptrs[source] == graph.ptrs[source + 1]) {
            continue;
        }

        timer_ker.start();
        brandes_backward(d_graph, level, d_distance, d_sigmas, d_delta);

        brandes_vertex_sum_kernel<<<int(graph.num_virtual_vertices / NUM_THREADS) + 1,
                                    NUM_THREADS>>>(source, graph.num_vertices, d_distance, d_delta,
                                                   d_sigmas, d_betweenness);
        timer_ker.stop();
        elapsed_ker += timer_ker.elapsed();
    }

    HANDLE_ERROR(cudaMemcpy(h_betweenness, d_betweenness, graph.num_vertices * sizeof(double),
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(d_distance));
    HANDLE_ERROR(cudaFree(d_sigmas));
    HANDLE_ERROR(cudaFree(d_delta));
    HANDLE_ERROR(cudaFree(d_cont));
    HANDLE_ERROR(cudaFree(d_betweenness));

    timer_ker_mem.stop();
    elapsed_ker_mem = timer_ker_mem.elapsed();
    std::cerr << int(elapsed_ker) << std::endl;
    std::cerr << int(elapsed_ker_mem) << std::endl;

    return h_betweenness;
}
