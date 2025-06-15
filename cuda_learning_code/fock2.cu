#include <cuda.h>
#include <cstdio>

#include "input.cuh"

#define WARP_SIZE 32
#define FULLMASK 0xffffffff


__global__
void matrixTransposeInPlace(double* g_matrix, short num_CGTOs)
{
    if (blockIdx.x < blockIdx.y) {
        return;
    }
    const short xid = blockDim.x * blockIdx.x + threadIdx.x;
    const short yid = blockDim.y * blockIdx.y + threadIdx.y;
    if (xid < yid || xid >= num_CGTOs || yid >= num_CGTOs) {
        return;
    }

    //__shared__ double s_src[WARP_SIZE][WARP_SIZE];
    //__shared__ double s_dst[WARP_SIZE][WARP_SIZE];
    __shared__ double s_src[WARP_SIZE][WARP_SIZE + 1];
    __shared__ double s_dst[WARP_SIZE][WARP_SIZE + 1];
    s_src[threadIdx.y][threadIdx.x] = g_matrix[num_CGTOs * yid + xid];
    s_dst[threadIdx.y][threadIdx.x] = g_matrix[num_CGTOs * xid + yid];

    __syncthreads();

    g_matrix[num_CGTOs * yid + xid] = s_dst[threadIdx.y][threadIdx.x];
    g_matrix[num_CGTOs * xid + yid] = s_src[threadIdx.y][threadIdx.x];
}


__device__
long long ijkl2serial(short i, short j, short k, short l, int num_CGTOs)
{
    long long eid = num_CGTOs * num_CGTOs * num_CGTOs * i + \
                    num_CGTOs * num_CGTOs * j + \
                    num_CGTOs * k + \
                    l;
    return eid;
}


__global__
void init2zero(double* g_fock, short num_CGTOs)
{
    const int num_square = num_CGTOs * num_CGTOs;
    const int pid = blockDim.x * blockDim.y * blockIdx.x + \
                    blockDim.x * threadIdx.y + threadIdx.x;
    if (pid >= num_square) {
        return;
    }
    const short i = pid / num_CGTOs;
    const short j = pid % num_CGTOs;
    g_fock[num_CGTOs * i + j] = 0.0;
}


__global__
void init2zero_sq(double* g_fock, short num_CGTOs)
{
    const int num_square = num_CGTOs * num_CGTOs;
    const int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_square) {
        return;
    }
    const short i = pid / num_CGTOs;
    const short j = pid % num_CGTOs;
    g_fock[num_CGTOs * i + j] = 0.0;
}

__global__
void init2zero_pair(double* g_matrix, short num_CGTOs)
{
    const int num_square = num_CGTOs * num_CGTOs;
    const int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_square) {
        return;
    }
    const short i = pid / num_CGTOs;
    const short j = pid % num_CGTOs;
    g_matrix[num_CGTOs * i + j] = 0;
}

/*
__global__
void init2zero_quartet(double* g_matrix, short num_CGTOs)
{
    const int num_quad = num_CGTOs * num_CGTOs * num_CGTOs * num_CGTOs;
    const int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_square) {
        return;
    }
    const short i = pid / num_CGTOs;
    const short j = pid % num_CGTOs;
    g_matrix[num_CGTOs * i + j] = 0;
}
/**/


//*
__global__
void cudaCalculateEnergy(double* g_D, double* g_H, double* g_F, double* g_rhfE, 
                         short num_CGTOs)
{
//    if (g_rhfE[0] != 0.0 && blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//        printf("g_rhfE[0]: %f\n", g_rhfE[0]);
//    }

    const int num_square = num_CGTOs * num_CGTOs;
    const int pid = blockDim.x * blockDim.y * blockIdx.x + \
                    blockDim.x * threadIdx.y + threadIdx.x;
    /*
    if (pid >= num_square) {
        return;
    }
    /**/

    __shared__ double s_sigma[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_sigma[0] = 0.0;
    }
    __syncthreads();

    double sigma;
    if (pid < num_square) {
        sigma = g_D[pid] * (g_H[pid] + g_F[pid]);
    } else {
        sigma = 0.0;
    }

    //double sigma = g_D[pid] * (g_H[pid] + g_F[pid]);
    for (int offset = 16; offset > 0; offset /= 2) {
        sigma += __shfl_down_sync(FULLMASK, sigma, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_sigma, sigma);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_rhfE, s_sigma[0]);
    }
}
/**/


__global__
void deviceExpandEvalues1Dto2D(double* g_w_1d, double* g_w_2d, short num_CGTOs)
{
    const short cid = blockDim.x * blockIdx.x + threadIdx.x;
    if (cid >= num_CGTOs) {
        return;
    }
    g_w_2d[num_CGTOs * cid + cid] = g_w_1d[cid];
}

__global__
void deviceDivideSquareRoot(double* g_w_2d, short num_CGTOs)
{
    const short cid = blockDim.x * blockIdx.x + threadIdx.x;
    if (cid >= num_CGTOs) {
        return;
    }
    const int did = num_CGTOs * cid + cid;
    g_w_2d[did] = 1 / __dsqrt_rn(g_w_2d[did]);
}

__global__
void cudaDensityMatrix(double* g_coeff, double* g_dens, double alpha, 
                       int num_electrons, short num_CGTOs)
{
    const int num_square = num_CGTOs * num_CGTOs;
    const int pid = blockDim.x * blockDim.y * blockIdx.x + \
                    blockDim.x * threadIdx.y + threadIdx.x;
    if (pid >= num_square) {
        return;
    }

    const short mu = pid / num_CGTOs;
    const short nu = pid % num_CGTOs;

    double sigma = 0.0;
    for (int e = 0; e < num_electrons / 2; ++e) {
        sigma += g_coeff[num_CGTOs * mu + e] * g_coeff[num_CGTOs * nu + e];
    }
    g_dens[pid] = (1 - alpha) * 2 * sigma + alpha * g_dens[pid];
}


/*
__global__
void findDampingParams(double* g_Dn, double* g_Do, double* g_Fn, double* g_Fo, 
                       double* g_s, double* g_c, short nao)
{
    const int sqnao = nao * nao;
    const int pid = blockDim.x * blockDim.y * blockIdx.x + \
                    blockDim.x * threadIdx.y + threadIdx.x;
    if (pid >= sqnao) {
        return;
    }   


}
/**/

__global__
void traceMatrix(double* g_matrix, double* g_sc, int nao)
{
    __shared__ double s_sc;
    if (threadIdx.x == 0) {
        s_sc = 0;
    }
    __syncthreads();

    atomicAdd(&s_sc, g_matrix[nao * threadIdx.x + threadIdx.x]);
    __syncthreads();
    if (threadIdx.x == 0) {
        g_sc[0] = s_sc;
    }
}


__global__
void squareFock(double* g_fock, double* g_eri, double* g_dens, double* g_H,
                short num_CGTOs)
{
    const int bra = blockIdx.x;
    const short i = bra / num_CGTOs;
    const short j = bra % num_CGTOs;

    // 2-fold symmetry (vertical)
    /*
    const short j = __double2int_rn((__dsqrt_rn(8 * bra + 1) - 1) / 2);
    const short i = bra - j * (j + 1) / 2;
    const int uid = num_CGTOs * i + j;
    const int lid = num_CGTOs * j + i;
    /**/

    const short l = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ double s_F_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_F_ij[0] = 0.0;
    }
    __syncthreads();

    double sigma = 0.0;
    long long eid1, eid2;
    if (l < num_CGTOs) {
        for (short k = 0; k < num_CGTOs; ++k) {
            eid1 = ijkl2serial(i, j, k, l, num_CGTOs);
            //eid2 = ijkl2serial(i, l, k, j, num_CGTOs);
            eid2 = ijkl2serial(i, k, j, l, num_CGTOs);
            sigma += (g_eri[eid1] - 0.5 * g_eri[eid2]) * g_dens[num_CGTOs * k + l];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sigma += __shfl_down_sync(FULLMASK, sigma, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_F_ij, sigma);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        g_fock[bra] = s_F_ij[0] + g_H[bra];
        //g_fock[uid] = g_fock[lid] = s_F_ij[0] + g_H[uid];   // 2-fold symmetry
        //g_fock[bra] = s_F_ij[0];  // use cuBLAS
    }
}


/*
__global__
void squareFock(double* g_fock, double* g_eri, double* g_dens, double* g_H,
                short num_CGTOs)
{
    //const int num_square = num_CGTOs * num_CGTOs;
    const int bra = blockIdx.x;
    const short i = bra / num_CGTOs;
    const short j = bra % num_CGTOs;
    const short k = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ double s_F_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_F_ij[0] = 0.0;
    }
    __syncthreads();

    double sigma = 0.0;
    long long eid1, eid2;
    if (k < num_CGTOs) {
        for (short l = 0; l < num_CGTOs; ++l) {
            eid1 = ijkl2serial(i, j, k, l, num_CGTOs);
            eid2 = ijkl2serial(i, l, k, j, num_CGTOs);
            sigma += (g_eri[eid1] - 0.5 * g_eri[eid2]) * g_dens[num_CGTOs * k + l];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sigma += __shfl_down_sync(FULLMASK, sigma, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_F_ij, sigma);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        g_fock[bra] = s_F_ij[0] + g_H[bra];  // gpurhf
        //g_fock[bra] = s_F_ij[0];  // curhf
    }
}
/**/





__global__
void quadFock(double* g_fock, double* g_eri, double* g_dens, short num_CGTOs)
{
    const long long num_quads = num_CGTOs * num_CGTOs * num_CGTOs * num_CGTOs;
    const long long qid = blockDim.x * blockDim.y * blockIdx.x + \
                          blockDim.x * threadIdx.y + threadIdx.x;
    if (qid >= num_quads) {
        //printf("qid: %ld\n", qid);
        return;
    }

    const int bra = qid / (num_CGTOs * num_CGTOs);
    const int ket = qid % (num_CGTOs * num_CGTOs);
    const short i = bra / num_CGTOs;
    const short j = bra % num_CGTOs;
    const short k = ket / num_CGTOs;
    const short l = ket % num_CGTOs;

    /*
    if (qid == 1666) {
        printf("bra: %d\n", bra);
        printf("ket: %d\n", ket);
    }
    /**/

    const long long eid1 = ijkl2serial(i, j, k, l, num_CGTOs);
    const long long eid2 = ijkl2serial(i, l, k, j, num_CGTOs);
    const double F_ijkl = (g_eri[eid1] - 0.5 * g_eri[eid2]) * g_dens[ket];

    //g_eri[ijkl2serial(i, j, k, l, num_CGTOs)] = F_ijkl;
    /*
    if (i == 3 && j == 1 && k == 0 && l == 6) {
        printf("F_ij: %f\n", F_ijkl);
    }
    /**/

    //printf("%d, %f\n", bra, g_fock[bra]);
    
    atomicAdd(g_fock + bra, F_ijkl);
    //atomicAdd(&g_fock[bra], F_ijkl);
}

//DIIS

__global__
void createBblock(double** d_B, double **d_e_list, short size, short num_CGTOs) 
{
    const int tid = blockDim.x * threadIdx.y + threadIdx.x;
    double er;
    double sum = 0;
    __shared__ double local_sum[1];

    if (blockIdx.x < size && blockIdx.y < size)
    {
        if (threadIdx.x < num_CGTOs && threadIdx.y < num_CGTOs)
        {
            sum = d_e1[tid] * d_e2[tid];
            atomicAdd(&local_sum[0], sum);
        }
        __syncthreads();
        d_B[blockIdx.y][blockIdx.x] = local_sum[0];
    }else if(blockIdx.x == size || blockIdx.y == size)
    {
        d_B[blockIdx.y][blockIdx.x] = -1;
        if(blockIdx.x == size && blockIdx.y == size)
        {
            d_B[blockIdx.y][blockIdx.x] = 0;
        }
    }

}

