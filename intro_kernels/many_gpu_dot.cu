#include <stdio.h>
#include "cuda.h"


struct DataStruct {
    int deviceId; 
    int size; 
    float *a; 
    float *b; 
    float returnValue; 
};


// Split work up between 2 GPus, and run kernels from different CPU threads. 
int main(void) {
    int deviceCount; 
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        printf("Need at least 2 GPUs\n");
        return 1;
    }

    float *a = (float*)malloc(sizeof(float) * N); 
    float *b = (float*)malloc(sizeof(float) * N); 

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    DataStruct data[2]; 
    data[0].deviceId = 0; 
    data[0].size = N; 
    data[0].a = a; 
    data[0].b = b; 
    data[0].returnValue = 0; 

    data[1].deviceId = 1; 
    data[1].size = N; 
    data[1].a = a + N/2; 
    data[1].b = b + N/2; 
    data[1].returnValue = 0; 

    // start threads and execute, then reduce to answer. 
}