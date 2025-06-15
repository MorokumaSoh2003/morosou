#include <stdio.h>
#include <cuda.h>

__global__ void matMul(float *A, float *B, float *C, int N) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float Cvalue = 0;
    for (int i = 0; i < N; i++) {
        Cvalue += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = Cvalue;
}

int main() {

    //配列のサイズ
    int N = 1024;
    size_t size = N * N * sizeof(float);

    //CPU側の配列確保
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    for(int k=0; k < N*N; k++) {
        A[k] = 1;
        B[k] = 1;
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
    cudaEvent_t start, stop;
    float e_time = 0.0;
    dim3 dimGrid(N/32, N/32);
    dim3 dimBlock(32, 32);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); // timer start
    matMul<<<dimGrid, dimBlock>>>(g_A, g_B, g_C, N);
    cudaEventRecord(stop,0); // timer stop
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&e_time, start, stop);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // Read C from device memory
    cudaMemcpy(C, g_C, size, cudaMemcpyDeviceToHost);

    printf("\tElapsed time: %f [ms]\n",e_time);

    int flag = 0;

    for(int i=0; i < N*N; i++) {
        if(C[i] != N) {
            flag = 1;
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