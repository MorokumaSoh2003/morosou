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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int *A, *B, *C;
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    cudaHostAlloc((void**)&A, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&B, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&C, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for(int i=0; i<FULL_DATA_SIZE; i++) {
        A[i] = 1;
        B[i] = 1;
    }

    cudaEventRecord(start, 0);

    for(int i=0; i<FULL_DATA_SIZE; i+=N) {
        cudaMemcpyAsync(dev_a, A+i, N*sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, B+i, N*sizeof(int), cudaMemcpyHostToDevice, stream);

        kernel<<<N/256, 256, 0, stream>>>(dev_a, dev_b, dev_c);

        cudaMemcpyAsync(C+i, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);

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
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaStreamDestroy(stream);
    
    return 0;
}