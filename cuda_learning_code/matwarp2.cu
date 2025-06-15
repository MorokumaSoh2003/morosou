#include <stdio.h>
#include <cuda.h>

__global__ void MatAdd(float *A, float *B, float *C, int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < N) {
        if(threadIdx.x%2  == 0) {
            C[index] = A[index] + B[index];
        }else {
            C[index] = A[index] - B[index];
        }
    }
}

int main() {

    //配列のサイズ
    int N = 10000;
    size_t size = N * sizeof(float);

    //CPU側の配列確保
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    for(int i=0; i<N; i++) {
        A[i] = 1;
        B[i] = 1;
    }

    //GPU側の配列
    float *g_A, *g_B, *g_C;
    cudaMalloc((void**) &g_A, size);
    cudaMalloc((void**) &g_B, size);
    cudaMalloc((void**) &g_C, size);

    //CPUの配列をGPUにコピー
    cudaMemcpy(g_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_C, C, size, cudaMemcpyHostToDevice);

    //カーネル実行
    int threadPerBlock = 256;
    int blockPerGrid = (N + threadPerBlock - 1)/ threadPerBlock;
    cudaEvent_t start, stop;
    float e_time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); // timer start
    MatAdd<<<threadPerBlock, blockPerGrid>>>(g_A, g_B, g_C, N);
    cudaEventRecord(stop,0); // timer stop
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&e_time, start, stop);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // Read C from device memory
    cudaMemcpy(C, g_C, size, cudaMemcpyDeviceToHost);

    printf("\tElapsed time: %f [ms]\n",e_time);

    int flag = 0;
    for(int i=0; i<N; i++) {
        if(i%2 == 0) {
            if(C[i] != 2) flag = 1; 
        }else {
            if(C[i] != 0) flag = 1; 
        }
    }

    if(flag == 0) {
        printf("answer is true.\n");
    } else {
        printf("answer is false.\n");
    }

    //GPUメモリの解放
    cudaFree(g_A);
    cudaFree(g_B);
    cudaFree(g_C);

    //CPUメモリの解放
    free(A);
    free(B);
    free(C);

}