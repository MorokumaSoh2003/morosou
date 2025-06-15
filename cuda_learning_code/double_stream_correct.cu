#include <stdio.h>
#include <cuda.h>
#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int *A, int *B, int *C) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < N) {
        int index1 = (index + 1) % 256;
        int index2 = (index + 2) % 256;
        float as = (A[index] + A[index1] + A[index2]) /3.0f;
        float bs = (B[index] + B[index1] + B[index2]) /3.0f;

        C[index] = (as + bs) / 2;

    }
}

int main() {
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if(!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no speed up from streams\n");

        return 0;
    }

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate( &start);
    cudaEventCreate( &stop);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);


    int *A, *B, *C;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;

    cudaMalloc((void**)&dev_a0, N * sizeof(int));
    cudaMalloc((void**)&dev_b0, N * sizeof(int));
    cudaMalloc((void**)&dev_c0, N * sizeof(int));

    cudaMalloc((void**)&dev_a1, N * sizeof(int));
    cudaMalloc((void**)&dev_b1, N * sizeof(int));
    cudaMalloc((void**)&dev_c1, N * sizeof(int));

    cudaHostAlloc((void**)&A, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&B, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&C, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for(int i=0; i<FULL_DATA_SIZE; i++) {
        A[i] = 1;
        B[i] = 1;
    }

    cudaEventRecord(start, 0);

    for(int i=0; i<FULL_DATA_SIZE; i+=N*2) {
        cudaMemcpyAsync(dev_a0, A+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_a1, A+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);

        cudaMemcpyAsync(dev_b0, B+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b1, B+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);

        kernel<<<N/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
        kernel<<<N/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

        cudaMemcpyAsync(C+i, dev_c0, N*sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(C+i+N, dev_c1, N*sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    printf("\tElapsed time: %f [ms]\n",elapsedTime);

    int flag = 0;

    for(int i=0; i < FULL_DATA_SIZE; i++) {
        if(C[i] != 1) {
            
            flag = 1;
        }
    }

    if(flag == 0) {
        printf("answer is true.\n");
    } else {
        printf("answer is false.\n");
    }

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    
    return 0;
}