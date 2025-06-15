#include <stdio.h>
#include <cuda.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

__device__ void MatTrans(float* in, float* out, int width, int height) {
    __shared__ float sh_mem[32][33];
    int src_block = blockIdx.y*32*height + blockIdx.x*32;
    int dst_block = blockIdx.y*32*width + blockIdx.x*32;

    sh_mem[threadIdx.y][threadIdx.x] = in[src_block + threadIdx.y*width + threadIdx.x];
    __syncthreads();
    out[dst_block + threadIdx.y*width + threadIdx.x] = sh_mem[threadIdx.x][threadIdx.y];
}


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C, Matrix B_T) {
    MatTrans(B.elements, B_T.elements, A.width, A.height);
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int e = 0; e < A.width; ++e) {
         Cvalue += A.elements[row * A.width + e] * B_T.elements[row * A.width + e];
    }
    C.elements[row * C.width + col] = Cvalue;
}

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A;
    d_A.width = A.width; 
    d_A.height = A.height;
    
    Matrix d_B;
    d_B.width = B.width; 
    d_B.height = B.height;
    
    Matrix d_C;
    d_C.width = C.width; 
    d_C.height = C.height;

    Matrix d_B_T;
    d_B_T.width = A.width; 
    d_B_T.height = A.height;

    size_t size = A.width * A.height * sizeof(float);

    cudaMalloc(&d_A.elements, size);
    cudaMalloc(&d_B.elements, size);
    cudaMalloc(&d_C.elements, size);
    cudaMalloc(&d_B_T.elements, size);

    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);



    cudaEvent_t start, stop;
    float e_time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); // timer start
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_B_T);
    cudaEventRecord(stop,0); // timer stop
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&e_time, start, stop);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    printf("\tElapsed time: %f [ms]\n",e_time);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

int main() {
    //配列のサイズ
    Matrix A, B, C;

    int N = 1024;

    A.width = N;
    A.height = N;
    size_t size_A = A.width * A.height * sizeof(float);

    B.width = N;
    B.height = N;
    size_t size_B = A.width * A.height * sizeof(float);

    C.width = N;
    C.height = N;
    size_t size_C = A.width * A.height * sizeof(float);

    A.elements = (float*)malloc(size_A);
    B.elements = (float*)malloc(size_B);
    C.elements = (float*)malloc(size_C);


    for (int i = 0; i < N*N; i++) {
        A.elements[i] = 1;
        B.elements[i] = 1;
        C.elements[i] = 0;
    }
    
    MatMul(A, B, C);

    free(A.elements);
    free(B.elements);
    free(C.elements);
}
