#include <stdio.h>
#include <cuda.h>

#define sh_size 32

__global__ void matMul(int *A, int *B, int N) {
    __shared__ float sh_mem[sh_size][sh_size+1];
    int src_block = blockIdx.y*sh_size*N + blockIdx.x*sh_size;
    int dst_block = blockIdx.x*sh_size*N + blockIdx.y*sh_size;

    sh_mem[threadIdx.y][threadIdx.x] = A[src_block + threadIdx.y*N + threadIdx.x];
    __syncthreads();
    B[dst_block + threadIdx.y*N + threadIdx.x] = sh_mem[threadIdx.x][threadIdx.y];   
}

int main() {

    //配列のサイズ
    int N = 1024;
    size_t size = N * N * sizeof(int);

    //CPU側の配列確保
    int *A = (int*)malloc(size);
    int *B = (int*)malloc(size);
    
    for(int i=0; i < N; i++) {
        for(int j=0; j< N; j++) {
            A[i*N + j] = i*N + j;
        }
    }

    //GPU側の配列
    int *g_A, *g_B;
    cudaMalloc((void**) &g_A, size);
    cudaMalloc((void**) &g_B, size);

    //CPUの配列をGPUにコピー
    cudaMemcpy(g_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_B, B, size, cudaMemcpyHostToDevice);

    //カーネル実行
    cudaEvent_t start, stop;
    float e_time = 0.0;
    dim3 dimGrid(N/32, N/32);
    dim3 dimBlock(32, 32);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); // timer start
    matMul<<<dimGrid, dimBlock>>>(g_A, g_B, N);
    cudaEventRecord(stop,0); // timer stop
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&e_time, start, stop);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // Read C from device memory
    cudaMemcpy(B, g_B, size, cudaMemcpyDeviceToHost);

    printf("\tElapsed time: %f [ms]\n",e_time);

    int flag = 0;

    for(int i=0; i < N; i++) {
        for(int j=0; j< N; j++) {
            //printf("%f ", B[i*N + j]);
            //printf("%f ", A[j*N + i]);
            if(B[i*N + j] != j*N + i) flag = 1;
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

    //CPUメモリの解放
    free(A);
    free(B);
}