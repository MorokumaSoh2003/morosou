#include <stdio.h>
#include <cuda.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 32

__device__ void MatTrans(float* in, float* out, int width, int height) {
    __shared__ float sh_mem[32][33];
    int src_block = blockIdx.y*32*height + blockIdx.x*32;
    int dst_block = blockIdx.y*32*width + blockIdx.x*32;

    sh_mem[threadIdx.y][threadIdx.x] = in[src_block + threadIdx.y*width + threadIdx.x];
    __syncthreads();
    out[dst_block + threadIdx.y*width + threadIdx.x] = sh_mem[threadIdx.x][threadIdx.y];
}


__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
        }
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
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

    size_t size = A.width * A.height * sizeof(float);

    cudaMalloc(&d_A.elements, size);
    cudaMalloc(&d_B.elements, size);
    cudaMalloc(&d_C.elements, size);

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
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
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

