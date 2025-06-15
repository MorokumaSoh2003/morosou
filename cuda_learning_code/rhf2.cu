#include <cstdio>
#include <chrono>
//#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <string>
#include <list>
#include <iostream>
#include <iterator>

#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

//#include "cblas.h"
//#include "lapacke.h"

//#include "rhf.cuh"
#include "input.cuh"
#include "gpurhf.cuh"
#include "device.cuh"
#include "moc.cuh"

#define listsize 5

std::vector<short> countEachOrbitalNumber(PrimitiveShell* h_ps, 
                                          short nps, int8_t orbitals)
{
    std::vector<short> num_orbital(orbitals, 0);
    for (short i = 0; i < nps; ++i) {
        num_orbital[h_ps[i].orbital]++;
    }
    //printf("num_orbital[0]: %d\n", num_orbital[0]);
    //printf("num_orbital[1]: %d\n", num_orbital[1]);
    return num_orbital;
}

std::vector<int> countEachShellPairNumber(const std::vector<short>& num_orbital)
{
    const int8_t orbitals = num_orbital.size();
    short tid = 0;
    std::vector<int> num_eachpair(orbitals * (orbitals + 1) / 2, 0);
    for (int8_t La = 0; La < orbitals; ++La) {
        for (int8_t Lb = La; Lb < orbitals; ++Lb) {
            if (La == Lb) {
                num_eachpair[tid] = num_orbital[La] * (num_orbital[La] + 1) / 2;
            }
            else {
                num_eachpair[tid] = num_orbital[La] * num_orbital[Lb];
            }
            tid++;
        }
    }
    //printf("num_eachpair[0]: %d\n", num_eachpair[0]);
    return num_eachpair;
}


double* runMolecularOrbitalMethod(Atom* h_nuc, short nnuc, short nao, 
                                  PrimitiveShell* h_ps, short nps, 
                                  double* h_C, double cutoff, double df)
{
    for (int i = 0; i < nnuc; ++i) {
        h_nuc[i].coord.x /= ANGSTROM_PER_BOHR;
        h_nuc[i].coord.y /= ANGSTROM_PER_BOHR;
        h_nuc[i].coord.z /= ANGSTROM_PER_BOHR;
    }

    for (short i = 0; i < nps; ++i) {
        h_ps[i].R.x /= ANGSTROM_PER_BOHR;
        h_ps[i].R.y /= ANGSTROM_PER_BOHR;
        h_ps[i].R.z /= ANGSTROM_PER_BOHR;
    }

    // hack
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    cusolverDnDestroy(cusolverH);

    int nelec = 0;
    for (short nid = 0; nid < nnuc; ++nid) {
        nelec += h_nuc[nid].charge;
    }
    const int sqnao = nao * nao;

    double* d_C;
    double* d_Eps;
    double* d_G;
    cudaMalloc(&d_C, sizeof(double) * sqnao);
    cudaMalloc(&d_Eps, sizeof(double) * sqnao);
    cudaMalloc(&d_G, sizeof(double) * sqnao * sqnao);
    cudaMemset(d_G, 0, sizeof(double) * sqnao * sqnao);
    cudaMemcpy(d_C, h_C, sizeof(double) * sqnao, cudaMemcpyHostToDevice);

    std::string scf = "stored";
    // Hartree-Fock method
    std::pair<double, float> result;
    result = rhf(h_nuc, nnuc, nao, nelec, h_ps, nps, df, cutoff, scf, d_C, d_Eps, d_G, h_C);



    /*
    // Integral transformation
    double* h_C;
    double* h_Eps;
    double* h_Gao;
    double* h_Gmo;
    cudaMallocHost(&h_C, sizeof(double) * sqnao);
    cudaMallocHost(&h_Eps, sizeof(double) * sqnao);
    cudaMallocHost(&h_Gao, sizeof(double) * sqnao * sqnao);
    cudaMallocHost(&h_Gmo, sizeof(double) * nao * nao * nao * nao);
    cudaMemcpy(h_C, d_C, sizeof(double) * sqnao, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Eps, d_Eps, sizeof(double) * sqnao, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Gao, d_G, sizeof(double) * sqnao * sqnao, cudaMemcpyDeviceToHost);
    //ao2mo(nao, h_C, h_Gao, h_Gmo);

     Møller–Plesset method
    double E_mp2 = 0.0;
    E_mp2 = mp2(nao, nelec, h_Gao, h_Eps, E_rhf);
    printf("E_mp2: %.12f\n", E_mp2);

    cudaFree(h_C);
    cudaFree(h_Eps);
    cudaFree(h_Gao);
    cudaFree(h_Gmo);
    /**/

    const double E_rhf = result.first;
    // device
    double* d_tmp;
    cudaMalloc(&d_tmp, sizeof(double) * sqnao * sqnao);
    //atom2mol(nao, d_C, d_G, d_tmp);
    //atom2mol4mp2(nao, nelec, d_C, d_G, d_tmp);
    //atom2mol4mp2_grouped(nao, nelec, d_C, d_G, d_tmp);
    //atom2mol_dgemm(nao, d_C, d_G, d_tmp);

    std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    ao2mo_dgemm(nao, d_C, d_G, d_tmp);
    stop = std::chrono::system_clock::now();
    auto dgemm_ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1e3;
    printf("AO2MO (DGEMM): %.3f [ms]\n", dgemm_ms);

    //cudaMemcpy(h_Gao, d_G, sizeof(double) * sqnao * sqnao, cudaMemcpyDeviceToHost);

    // device MP2
    double E_mp2 = mp2d(nao, nelec, d_G, d_Eps, E_rhf);
    printf("E_mp2: %.12f\n", E_mp2);
    cudaFree(d_tmp);



    cudaFree(d_G);
    cudaFree(d_C);
    cudaFree(d_Eps);

    return nullptr;
}





std::pair<double, float> rhf(Atom* h_nuc, short nnuc, short nao, int nelec, 
                             PrimitiveShell* h_ps, short nps, 
                             double df, double cutoff, std::string scf, 
                             double* d_C, double* d_Eps, double* d_G, double* h_C)
{
    const int sqnao = nao * nao;

    // Step 2. calculate molecular integrals
    double* d_S;
    double* d_H;
    cudaMalloc(&d_S, sizeof(double) * sqnao);
    cudaMalloc(&d_H, sizeof(double) * sqnao);
    cudaMemset(d_S, 0, sizeof(double) * sqnao);
    cudaMemset(d_H, 0, sizeof(double) * sqnao);

    Atom* d_atom;
    PrimitiveShell* d_pshell;
    cudaMalloc(&d_atom, sizeof(Atom) * nnuc);
    cudaMalloc(&d_pshell, sizeof(PrimitiveShell) * nps);

    double* d_F;
    double* d_Fp;
    double* d_D;
    double* d_X;
    cudaMalloc(&d_F, sizeof(double) * sqnao);
    cudaMalloc(&d_Fp, sizeof(double) * sqnao);
    cudaMalloc(&d_D, sizeof(double) * sqnao);
    cudaMalloc(&d_X, sizeof(double) * sqnao);
    cudaMemset(d_D, 0, sizeof(double) * sqnao);


    // Step 3.
    double* d_w_2d;
    cudaMalloc(&d_w_2d, sizeof(double) * sqnao);
    cudaMemset(d_w_2d, 0, sizeof(double) * sqnao);

    // generate the lookup table for the Boys function
    const size_t table_size = sizeof(double) * LUT_NUM_XI * (LUT_N_RANGE + LUT_K_MAX + 1);
    double* h_F_xi;
    double* d_F_xi;
    cudaMallocHost(&h_F_xi, table_size);
    cudaMalloc(&d_F_xi, table_size);
    //cudaMemset(h_F_xi, 0.0, table_size);
    generateTaylorTable(LUT_N_RANGE, LUT_K_MAX, LUT_XI_RANGE, LUT_XI_INTERVAL, h_F_xi);
    cudaMemcpy(d_F_xi, h_F_xi, table_size, cudaMemcpyHostToDevice);


    // hack
    //cusolverDnHandle_t cusolverH = NULL;
    //cusolverDnCreate(&cusolverH);
    //cusolverDnDestroy(cusolverH);

    std::chrono::system_clock::time_point rhf_s, rhf_e;
    rhf_s = std::chrono::system_clock::now();

    //std::sort(CGTO, CGTO + nao, compareContractedGTO);
    std::sort(h_ps, h_ps + nps, comparePrimitiveShell);
    const int8_t orbitals = h_ps[nps - 1].orbital + 1;

    std::vector<short> num_orbital = \
        countEachOrbitalNumber(h_ps, nps, orbitals);
    std::vector<int> num_eachpair = \
        countEachShellPairNumber(num_orbital);

    // Step 1. calculate nuclear replusion energy
    double nuclearE = nuclearRepulsionEnergy(h_nuc, nnuc);

    cudaMemcpy(d_atom, h_nuc, sizeof(Atom) * nnuc, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pshell, h_ps, sizeof(PrimitiveShell) * nps, cudaMemcpyHostToDevice);

    float onee = oneElectronIntegral(d_pshell, nps, d_S, d_H, d_atom, nnuc, nao, num_orbital, num_eachpair, d_F_xi);
    float twoe = twoElectronIntegral(h_ps, d_pshell, nps, d_G, nao, cutoff, num_orbital, num_eachpair, d_F_xi);
    //printf("1e integrals: %.3f [ms]\n", onee);

    deviceSymmetricDiagonalize(d_S, d_w_2d, nao);
    deviceTransformationMatrix(d_S, d_w_2d, d_X, nao);

    // Step 4. Initialize the density matrix P
    if (true) {
        deviceInitCoefficientMatrix(d_C, d_H, d_X, d_Eps, nao);
        //printf("initialized\n");
    }
    deviceBuildDensityMatrix(d_C, d_D, 0.0, nelec, nao);
    deviceBuildFockMatrix(d_F, d_G, d_D, d_H, nao);


    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    double convergedE = selfConsistentField(d_S, d_H, d_X, d_F, d_Fp, d_C, d_D, 
                                            d_G, d_Eps, nao, 
                                            nelec, cutoff, scf);
    end = std::chrono::system_clock::now();
    auto scf_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;

    printf("2e integrals: %.3f [ms]\n", twoe);
    printf("SCF interations: %.3f [ms]\n", scf_ms);

    //printf("nuclearE: %.8f\n", nuclearE);
    double rhfE = convergedE + nuclearE;


    rhf_e = std::chrono::system_clock::now();
    auto rhf_ms = std::chrono::duration_cast<std::chrono::microseconds>(rhf_e - rhf_s).count() / 1e3;

    printf("Restricted Hartree Fock: %.3f [ms]\n", rhf_ms);
    printf("RHF energy: %.12f [hartree]\n", rhfE);

    cudaFree(d_F);
    cudaFree(d_D);
    cudaFree(d_S);
    cudaFree(d_H);
    cudaFree(d_Fp);
    cudaFree(d_X);
    cudaFree(d_w_2d);
    cudaFree(d_F_xi);
    cudaFree(h_F_xi);
    cudaFree(d_atom);
    cudaFree(d_pshell);

    cudaMemcpy(h_C, d_C, sizeof(double) * sqnao, cudaMemcpyDeviceToHost);

    return std::make_pair(rhfE, rhf_ms);
}


double selfConsistentField(double* d_S, double* d_H, double* d_X, double* d_F, 
                           double* d_Fp, double* d_C, double* d_D, 
                           double* d_G, double* d_Eps, 
                           short nao, short nelec, 
                           double cutoff, std::string scf)
{
    short iteration = 0;
    double previousE = 0.0;
    double updatedE, deltaE;
    double* d_Cp = d_Fp;
    //double alpha = df;
    short max_iterations = 50;
    double* d_rhfE = nullptr;
    cudaMalloc(&d_rhfE, sizeof(double));

    // double df;
    const int sqnao = nao * nao;
    // double* d_swap;
    double* d_F_old;
    double* d_D_old;
    cudaMalloc(&d_F_old, sizeof(double) * sqnao);
    cudaMalloc(&d_D_old, sizeof(double) * sqnao);
    cudaMemset(d_F_old, 0, sizeof(double) * sqnao);
    cudaMemset(d_D_old, 0, sizeof(double) * sqnao);

    double* d_Dt;
    double* d_Ft;
    double* d_tmp;
    cudaMalloc(&d_Dt, sizeof(double) * nao * nao);
    cudaMalloc(&d_Ft, sizeof(double) * nao * nao);
    cudaMalloc(&d_tmp, sizeof(double) * nao * nao);

    double* d_e;
    cudaMalloc(&d_e, sizeof(double) * nao * nao); 
    std::vector<double> h_e(nao * nao);
    std::vector<double> h_F(nao * nao);
    cudaMemcpy(h_F.data(), d_F, sizeof(double) * nao * nao, cudaMemcpyDeviceToHost);

    std::vector<std::vector<double>> h_F_list; 
    h_F_list.insert(h_F_list.begin(), h_F);

    std::vector<std::vector<double>> h_e_list; 

    //*
    while (iteration < max_iterations) {
        updatedE = deviceCalculateEnergy(d_D, d_H, d_F, d_rhfE, nao);
        //printf("i = %d: %.8f\n", iteration, updatedE);
        deltaE = std::abs(updatedE - previousE);
        if (deltaE < 1.0e-10) {
            break;
        }
        previousE = updatedE;

        deviceOrthogonalize(d_F, d_X, d_Fp, nao);
        deviceSymmetricDiagonalize(d_Fp, d_Eps, nao);
        deviceUpdateCoefficientMatrix(d_X, d_Cp, d_C, nao);

        // d_swap = d_D;
        // d_D = d_D_old;
        // d_D_old = d_swap;
        // d_swap = d_F;
        // d_F = d_F_old;
        // d_F_old = d_swap;
        //cudaMemcpy(d_D_old, d_D, sizeof(double) * sqnao, cudaMemcpyDeviceToDevice);
        //cudaMemcpy(d_F_old, d_F, sizeof(double) * sqnao, cudaMemcpyDeviceToDevice);
        deviceBuildDensityMatrix(d_C, d_D, 0, nelec, nao);
        deviceBuildFockMatrix(d_F, d_G, d_D, d_H, nao);
        // df = optimizeDampingFactor4RHF(d_D, d_D_old, d_F, d_F_old, d_Dt, d_Ft, d_tmp, nao);
        // // df = 1;
        // linearInterpolateMatrix(d_D, d_D_old, nao, df);
        // linearInterpolateMatrix(d_F, d_F_old, nao, df);
        // printf("iteration %d: %.12f (alpha = %lf)\n", iteration, updatedE, df);
        calError(d_S, d_D, d_F, d_e, nao);
        // cudaMemcpy(h_e.data(), d_e, sizeof(double) * nao * nao, cudaMemcpyDeviceToHost);
        // std::copy(h_e.begin(), h_e.end(), std::ostream_iterator<double>(std::cout, " "));
        // std::cout << std::endl;
        pushList(d_F, d_e, h_F_list, h_e_list, nao);

        // std::vector<double> d_e_1d(nao * nao);
        // for (const auto& row : h_e_list) h_e_1d.insert(h_e_1d.end(), row.begin(), row.end());
        
        double **d_e_list;
        cudaMalloc((void**)&d_e_list, nao * nao * h_e_list.size() * sizeof(double));

        for (int i = 0; i < h_e_list.size(); i++)
        {
            cudaMemcpy(d_e_list[i], h_e_list[i].data(), sizeof(double) * nao * nao, cudaMemcpyHostToDevice);
        }
        cudaFree(d_e_list);
        break;
        iteration++;
    }
    /**/
    cudaFree(d_rhfE);
    cudaFree(d_F_old);
    cudaFree(d_D_old);
    cudaFree(d_Dt);
    cudaFree(d_Ft);
    cudaFree(d_tmp);
    
    

    if (iteration == max_iterations) {
        printf("SCF procedure has not been converged...\n");
        return 0.0;
    }
    else {
        printf("Successfully converged (#iteration: %d)\n", iteration);
        return updatedE;
    }
}


void linearInterpolateMatrix(double* d_new, double* d_old, int nao, double df)
{
    double alpha = 1 - df;
    double beta = df;
    cublasHandle_t cublasH = NULL;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasCreate(&cublasH);
    cublasDgeam(cublasH, transa, transb, nao, nao,
                &alpha, d_old, nao, &beta, d_new, nao, d_new, nao);
    cublasDestroy(cublasH);
}


double optimizeDampingFactor4RHF(double* d_Dn, double* d_Do, double* d_Fn, double* d_Fo, 
                                 double* d_Dt, double* d_Ft, double* d_tmp, short nao)
{
    double df, s, c;
    double* d_s;
    double* d_c;
    //double* d_Dt;
    //double* d_Ft;
    //double* d_tmp;
    cudaMalloc(&d_s, sizeof(double));
    cudaMalloc(&d_c, sizeof(double));
    //cudaMalloc(&d_Dt, sizeof(double) * nao * nao);
    //cudaMalloc(&d_Ft, sizeof(double) * nao * nao);
    //cudaMalloc(&d_tmp, sizeof(double) * nao * nao);
    cudaMemset(d_s, 0, sizeof(double));
    cudaMemset(d_c, 0, sizeof(double));

    double alpha = 1;
    double beta = -1;
    cublasHandle_t cublasH = NULL;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasCreate(&cublasH);
    // D_new - D_old
    cublasDgeam(cublasH, transa, transb, nao, nao,
                &alpha, d_Dn, nao, &beta, d_Do, nao, d_Dt, nao);
    // F_new - F_old
    cublasDgeam(cublasH, transa, transb, nao, nao,
                &alpha, d_Fn, nao, &beta, d_Fo, nao, d_Ft, nao);
    beta = 0;
    // F_old * (D_new - D_old)
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, 
                &alpha, d_Dt, nao, d_Fo, nao, &beta, d_tmp, nao);
    traceMatrix<<<1, nao>>>(d_tmp, d_s, nao);
    cudaDeviceSynchronize();
    // (F_new - F_old) * (D_new - D_old)
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, 
                &alpha, d_Dt, nao, d_Ft, nao, &beta, d_tmp, nao);
    traceMatrix<<<1, nao>>>(d_tmp, d_c, nao);
    cudaDeviceSynchronize();
    cublasDestroy(cublasH);

    cudaMemcpy(&s, d_s, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, d_c, sizeof(double), cudaMemcpyDeviceToHost);
    if (c > -0.5 * s) {
        df = -0.5 * s / c;
    }
    else {
        df = 1;
    }

    cudaFree(d_s);
    cudaFree(d_c);
    //cudaFree(d_Dt);
    //cudaFree(d_Ft);
    //cudaFree(d_tmp);

    return df;
}



void deviceUpdateCoefficientMatrix(double* d_X, double* d_Cp, double* d_C, 
                                   short nao) 
{
    //cudaMemset(d_C, 0, sizeof(double) * nao * nao);
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasCreate(&cublasH);
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, 
                &alpha, d_Cp, nao, d_X, nao, &beta, d_C, nao);
    cublasDestroy(cublasH);
}



double euclideanDistance(const Coordinate& P, const Coordinate& Q) {
    double xd = P.x - Q.x;
    double yd = P.y - Q.y;
    double zd = P.z - Q.z;
    double distance = std::sqrt(xd * xd + yd * yd + zd * zd);
    return distance;
}



double nuclearRepulsionEnergy(Atom* atom, short nnuc)
{
    double rAB;
    double nuclear_energy = 0.0;
    for (short A = 0; A < nnuc; ++A) {
        for (short B = A + 1; B < nnuc; ++B) {
            rAB = euclideanDistance(atom[A].coord, atom[B].coord);
            nuclear_energy += atom[A].charge * atom[B].charge / rAB;
        }
    }
    return nuclear_energy;
}






void deviceSymmetricDiagonalize(double* d_A, double* d_w_2d, short nao)
{
    double* d_w_1d = nullptr;
    int* d_info = nullptr;
    int lwork = 0;
    double* d_work = nullptr;

    const int division = (nao + 32 - 1) / 32;
    dim3 blocks(division, division);
    dim3 threads(32, 32);
    matrixTransposeInPlace<<<blocks, threads>>>(d_A, nao);

    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    cudaMalloc(&d_w_1d, sizeof(double) * nao);
    cudaMalloc(&d_info, sizeof(int));
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, nao, d_A, 
                                nao, d_w_1d, &lwork);
    cudaMalloc(&d_work, sizeof(double) * lwork);
    cusolverDnDsyevd(cusolverH, jobz, uplo, nao, d_A, nao, 
                     d_w_1d, d_work, lwork, d_info);

    matrixTransposeInPlace<<<blocks, threads>>>(d_A, nao);

    short num_blocks = (nao + 256 - 1) / 256;
    deviceExpandEvalues1Dto2D<<<num_blocks , 256>>>(d_w_1d, d_w_2d, nao);

    cudaFree(d_w_1d);
    cudaFree(d_info);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
}


void deviceTransformationMatrix(double* d_U, double* d_w_2d, double* d_X, 
                                short nao)
{
    short num_blocks = (nao + 256 - 1) / 256;
    deviceDivideSquareRoot<<<num_blocks , 256>>>(d_w_2d, nao);   

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasCreate(&cublasH);
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, 
                &alpha, d_w_2d, nao, d_U, nao, &beta, d_X, nao);
    
    cublasDestroy(cublasH);
}


void deviceOrthogonalize(double* d_target, double* d_trans, double* d_Fp, 
                         short nao)
{
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasCreate(&cublasH);
    cublasDgemm(cublasH, transa, CUBLAS_OP_T, nao, nao, nao, 
                &alpha, d_target, nao, d_trans, nao, 
                &beta, d_Fp, nao);
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, 
                &alpha, d_trans, nao, d_Fp, nao, 
                &beta, d_Fp, nao);
    cublasDestroy(cublasH);
}



void deviceInitCoefficientMatrix(double* d_C, double* d_H, double* d_X, 
                                 double* d_Eps, short nao)
{
    double* d_Hp;
    cudaMalloc(&d_Hp, sizeof(double) * nao * nao);
    double* d_Cp = d_Hp;
    deviceOrthogonalize(d_H, d_X, d_Hp, nao);
    deviceSymmetricDiagonalize(d_Hp, d_Eps, nao);
    deviceUpdateCoefficientMatrix(d_X, d_Cp, d_C, nao);
    cudaFree(d_Hp);
}


bool compareContractedGTO(const ContractedGTO& mu, const ContractedGTO& nu)
{
    short L_mu = mu.shell.l + mu.shell.m + mu.shell.n;
    short L_nu = nu.shell.l + nu.shell.m + nu.shell.n;
    if (L_mu != L_nu) {
        return L_mu < L_nu;
    }
    else {
        return mu.head < nu.head;
    }
}

bool comparePrimitiveShell(const PrimitiveShell& mu, const PrimitiveShell& nu)
{
    //*
    if (mu.orbital != nu.orbital) {
        return mu.orbital < nu.orbital;
    }
    else {
        return mu.cid < nu.cid;
    }
    /**/
    //return mu.orbital < nu.orbital;
}


void deviceBuildFockMatrix(double* d_F, double* d_G, 
                           double* d_D, double* d_H, 
                           short nao)
{
    const int8_t threadsPerWarp = 32;
    const int8_t warpsPerBlock = (nao + threadsPerWarp - 1) / threadsPerWarp;
    const short threadsPerBlock = threadsPerWarp * warpsPerBlock;
    if (threadsPerBlock > 1024) {
        printf("too many cgtos.\n");
        std::exit(EXIT_FAILURE);
    }
    const int num_blocks = nao * nao;
    //const int num_blocks = nao * (nao + 1) / 2;
    dim3 blocks(num_blocks);
    dim3 threads(threadsPerWarp, warpsPerBlock);
    squareFock<<<blocks, threads>>>(d_F, d_G, d_D, d_H, nao);

    /*
    cublasHandle_t cublasH = NULL;
    const double alpha = 1.0;
    cublasCreate(&cublasH);
    cublasDaxpy(cublasH, num_blocks, &alpha, d_F, 1, d_H, 1);
    cublasDestroy(cublasH);
    /**/
}


void deviceBuildDensityMatrix(double* d_C, double* d_D, double alpha, 
                              int nelec, short nao)
{
    const int sqnao = nao * nao;

    const int8_t threadsPerWarp = 32;
    const short threadsPerBlock = 256;
    const int8_t warpsPerBlock = threadsPerBlock / threadsPerWarp;
    const int num_blocks = (sqnao + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(num_blocks);
    dim3 threads(threadsPerWarp, warpsPerBlock);

    cudaDensityMatrix<<<blocks, threads>>>(d_C, d_D, alpha, nelec, nao);
}


double calculateRHFenergy(double* h_D, double* h_H, double* h_F, short nao)
{
    int cid;
    double sigma = 0.0;
    for (short mu = 0; mu < nao; ++mu) {
        for (short nu = 0; nu < nao; ++nu) {
            cid = nao * mu + nu;
            sigma += h_D[cid] * (h_H[cid] + h_F[cid]);
        }
    }
    return 0.5 * sigma;
}

//*
double deviceCalculateEnergy(double* d_D, double* d_H, double* d_F, 
                             double* d_rhfE, short nao) 
{
    const int sqnao = nao * nao;

    double rhfE = 0.0;
    cudaMemcpy(d_rhfE, &rhfE, sizeof(double), cudaMemcpyHostToDevice);

    const int8_t threadsPerWarp = 32;
    const short threadsPerBlock = 1024;
    const int8_t warpsPerBlock = threadsPerBlock / threadsPerWarp;
    const int num_blocks = (sqnao + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(num_blocks);
    dim3 threads(threadsPerWarp, warpsPerBlock);
    cudaCalculateEnergy<<<blocks, threads>>>(d_D, d_H, d_F, d_rhfE, nao);

    cudaMemcpy(&rhfE, d_rhfE, sizeof(double), cudaMemcpyDeviceToHost);

    //printf("%.8f\n", rhfE);

    return 0.5 * rhfE;
}
/**/

//DIIS
void calError(double* d_S, double* d_D, double* d_F, double* d_e, short nao)
{
    double* d_tmp;
    cudaMalloc((void **)&d_tmp, sizeof(double) * nao * nao);
    
    const double alpha = 1.0;
    double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasCreate(&cublasH);

    //F * P * S
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, &alpha, d_F, nao, d_D, nao, &beta, d_tmp, nao);
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, &alpha, d_tmp, nao, d_S, nao, &beta, d_e, nao);

    //S * P * F
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, &alpha, d_S, nao, d_D, nao, &beta, d_tmp, nao);
    beta = -1.0;
    cublasDgemm(cublasH, transa, transb, nao, nao, nao, &alpha, d_tmp, nao, d_F, nao, &beta, d_e, nao);

}

void pushList(double* d_F, double* d_e, std::vector<std::vector<double>> h_F_list, std::vector<std::vector<double>> h_e_list, short nao)
{
    std::vector<double> h_F(nao * nao);
    std::vector<double> h_e(nao * nao);
    cudaMemcpy(h_F.data(), d_F, sizeof(double) * nao * nao, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_e.data(), d_e, sizeof(double) * nao * nao, cudaMemcpyDeviceToHost);

    if(h_F_list.size() == listsize) h_F_list.pop_back();
    if(h_e_list.size() == listsize) h_e_list.pop_back();
    h_F_list.insert(h_F_list.begin(), h_F);
    h_e_list.insert(h_e_list.begin(), h_e);
}

// void calLambda(double** d_B, double* C, double* A, short size)
// {
//     int* d_pivot;
//     int* d_info;
//     double* d_work;
//     cusolverDnHandle_t handle;
//     cusolverDnCreate(&handle);
//     cudaMalloc((void**)&d_pivot, size * sizeof(int));
//     cudaMalloc((void**)&d_info, sizeof(int));

//     int workSize = 0;
//     cusolverDnDgetrf_bufferSize(handle, size, size, d_B, size, &workSize);
//     cudaMalloc((void**)&d_work, workSize * sizeof(double));

//     cusolverDnDgetrf(handle, size, size, d_B, size, d_work, d_pivot, d_info);

//     cusolverDnDgetrs(handle, CUBLAS_OP_N, size, 1, d_B, size, d_pivot, A, size, d_info);
//     cudaMemcpy(C, d_B, size * sizeof(double), cudaMemcpyDeviceToHost);
// }

// __global__
// void newFock(double** d_F_list, double* d_F, double* C, short size, short num_CGTOs)
// {
//     const int tid = blockDim.x * threadIdx.y + threadIdx.x;
//     double ;
    
//     for (int i = 0; i < size; i++)
//     {
//        d_F[tid] += d_F_list[i][tid] * C[size];
//     }
// }