#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <iostream>

__global__ void MatAdd2d(float *A, float *B, float *C, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = x + y * gridDim.x * blockDim.x;

    if(x < N || y < N) {
        C[index] = A[index] + B[index];
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

    auto start = std::chrono::high_resolution_clock::now();

    MatAdd2d<<<threadPerBlock, blockPerGrid>>>(g_A, g_B, g_C, N);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << duration.count() << "[ms]\n";

    cudaMemcpy(C, g_C, size, cudaMemcpyDeviceToHost);

    //GPUメモリの解放
    cudaFree(g_A);
    cudaFree(g_B);
    cudaFree(g_C);

    //CPUメモリの解放
    free(A);
    free(B);
    free(C);
}