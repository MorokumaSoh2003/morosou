#include <stdio.h>
#include <cuda.h>
#include <unistd.h>

__global__ void Sum(float *A, float *sum, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float local_sum[1];

    if(id < N){
        atomicAdd(&local_sum[0], A[id]);
    }
    __syncthreads();

    if(threadIdx.x == 0) {
        atomicAdd(sum, local_sum[0]);
    }
}

int main() {

    //配列のサイズ
    int N = 1024;
    size_t size = N * sizeof(float);

    //CPU側の配列確保
    float *A = (float*)malloc(size);
    float sum = 0;

    for(int k=0; k < N; k++) {
        A[k] = 1;
    }

    //GPU側の配列
    float *g_A, *g_sum;
    cudaMalloc((void**) &g_A, size);
    cudaMalloc((void**) &g_sum, sizeof(float));


    cudaMemset(g_sum, 0, sizeof(float));

    //CPUの配列をGPUにコピー
    cudaMemcpy(g_A, A, size, cudaMemcpyHostToDevice);


    //カーネル実行
    cudaEvent_t start, stop;
    float e_time = 0.0;
    int dimBlock = 32;
    int dimGrid = (N + dimBlock - 1)/ dimBlock;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); // timer start
    Sum<<<dimGrid, dimBlock>>>(g_A, g_sum, N);
    cudaEventRecord(stop,0); // timer stop
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&e_time, start, stop);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // Read C from device memory
    cudaMemcpy(&sum, g_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("\tElapsed time: %f [ms]\n",e_time);

    int flag = 0;

    printf("%f\n", sum);

    if(sum != N) {
        flag = 1;
    }

    if(flag == 0) {
        printf("answer is true.\n");
    } else {
        printf("answer is false.\n");
    }

    //GPUメモリの解放
    cudaFree(g_A);
    cudaFree(g_sum);

    //CPUメモリの解放
    free(A);

}