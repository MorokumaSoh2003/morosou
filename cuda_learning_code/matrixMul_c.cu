#include <stdio.h>
#include <chrono>
#include <iostream>

void matMul(float *A, float *B, float *C, int N) {
    float Cvalue = 0;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Cvalue = 0;

            for(int k=0; k < N; k++) {
                Cvalue += A[j*N + k] * B[k*N + i];
            }

            C[j*N + i] = Cvalue;
        }
    }
}

int main() {

    //配列のサイズ
    int N = 2048;
    size_t size = N * N * sizeof(float);

    //CPU側の配列確保
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    for(int k=0; k < N*N; k++) {
        A[k] = 1;
        B[k] = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    matMul(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << duration.count() << "[ms]\n";

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

    //CPUメモリの解放
    free(A);
    free(B);
    free(C);
}