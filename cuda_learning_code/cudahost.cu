#include <stdio.h>
#include <cuda.h>

#define SIZE (64*1024*1024)

float cuda_host_alloc_test(int size, bool up) {
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapesedTime;

    cudaEventCreate( &start);
    cudaEventCreate( &stop);

    cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault);

    cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

    cudaEventRecord(start, 0);
    for(int i=0; i<100; i++) {
        if(up) {
            cudaMemcpy(dev_a, a, sizeof(*a), cudaMemcpyHostToDevice);
        }else {
            cudaMemcpy(a, dev_a, sizeof(*a), cudaMemcpyDeviceToHost);
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapesedTime, start, stop);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    printf("\tElapsed time: %f [ms]\n",elapesedTime);

    cudaFreeHost(a);
    cudaFree(dev_a);

    return elapesedTime;
}

int main() {
    float elapesedTime;
    float MB = (float)100*SIZE*sizeof(int)/1024/1024;

    elapesedTime = cuda_host_alloc_test(SIZE, true);
    printf("Time using cudaMalloc: %3.1f ms\n", elapesedTime);
    printf("\t MB/s during copy up: %3.1f\n", MB/(elapesedTime/1000));
}