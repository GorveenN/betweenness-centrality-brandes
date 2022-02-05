#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

static int NUM_THREADS = 128;

#include <stdio.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

template <typename T>
__global__ void cudaMemcpyTypedKernel(T* devPtr, const T val, const size_t nwords) {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; tidx < nwords; tidx += stride)
        devPtr[tidx] = val;
}

template <typename T>
void cudaMemcpyTyped(T* devPtr, const T val, const size_t nwords) {
    cudaMemcpyTypedKernel<<<int(nwords / NUM_THREADS) + 1, NUM_THREADS>>>(devPtr, val, nwords);
}

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

class CUDATimer {
   private:
    cudaEvent_t _start, _stop;

   public:
    CUDATimer() {
        HANDLE_ERROR(cudaEventCreate(&_start));
        HANDLE_ERROR(cudaEventCreate(&_stop));
    }

    void start() { HANDLE_ERROR(cudaEventRecord(_start, 0)); }

    void stop() {
        HANDLE_ERROR(cudaEventRecord(_stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(_stop));
    }

    float elapsed() {
        float time;
        HANDLE_ERROR(cudaEventElapsedTime(&time, _start, _stop));
        return time;
    }

    ~CUDATimer() {
        HANDLE_ERROR(cudaEventDestroy(_start));
        HANDLE_ERROR(cudaEventDestroy(_stop));
    }
};

#endif  // CUDA_UTILS_CUH
