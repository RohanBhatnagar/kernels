#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
  // int i = blockIdx.x;
  int i = threadIdx.x;
  if (i < 10) {
    c[i] = a[i] + b[i];
  }
}

__global__ void init(int *dev_a, int *dev_b, int N) {
  int i = blockIdx.x;
  if (i < N) {
    dev_a[i] = i;
    dev_b[i] = i + 1;
  }
}

int main(void) {
  int N = 10;
  int c[N];

  int *dev_a, *dev_b, *dev_c;

  cudaMalloc((void **)&dev_a, sizeof(int) * N);
  cudaMalloc((void **)&dev_b, sizeof(int) * N);
  cudaMalloc((void **)&dev_c, sizeof(int) * N);

  init<<<N, 1>>>(dev_a, dev_b, N); // avoid storing local copies. params to kernel must be dev ptrs.
  add<<<1,N>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    printf("c[%d] = %d\n", i, c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}