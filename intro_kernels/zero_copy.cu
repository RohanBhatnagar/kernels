#include <stdio.h>
#include "cuda.h"

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, N + threadsPerBlock - 1 / threadsPerBlock);


// kernel for dot product. 
__global__ void dot(float *a, float *b, float *c) {
    __shared__float cache[threadsPerBlock]; // shared memory 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x + gridDim.x;
    }
    cache[cacheIdx] = temp;
    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
    }
    if (cacheIdx == 0) {
        c[blockIdx.x] = cache[0];
    }
}

float cuda_host_alloc(int size) {
    cudaEvent_t start, stop; 
    float *a, *b, c, *partial_c; 
    float elapsedTime; 

    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped); 
    cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped); 
    cudaHostAlloc((void**)&partial_c, blocksPerGrid * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped); 

    for (int i = 0; i < 100; i++) {
        a[i] = i; 
        b[i] = i*2; 
    }

    float *dev_a, *dev_b, *dev_partial_c; 
    cudaHostGetDevicePointer((void**)&dev_a, a, 0); // addr on gpu. 
    cudaHostGetDevicePointer((void**)&dev_b, b, 0); 
    cudaHostGetDevicePointer((void**)&dev_partial_c, partial_c, 0); 

    cudaEventRecord(start, 0); 
    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c); 
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsedTime, start, stop); 
    
    c = 0; 
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i]; 
    }
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsedTime, start, stop); 

    cudaFreeHost(a); 
    cudaFreeHost(b); 
    cudaFreeHost(partial_c); 
    cudaFree(dev_a); 

    printf("c = %f\n", c);
    printf("elapsed time = %f ms\n", elapsedTime);

    return elapsedTime;
}

int main(void) {
    int size = 100;
    float elapsedTime = cuda_host_alloc(size);
    printf("elapsed time = %f ms\n", elapsedTime);
    return 0;
}