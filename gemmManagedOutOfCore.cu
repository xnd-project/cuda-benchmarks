#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include "common.hh"

int main(int argc, char *argv[])
{
    size_t N = DEFAULT_N;
    if (argc==2) N = (size_t)atoi(argv[1]);
    printf("N=%zd\n", N);

    clock_t start, end; 
    cublasHandle_t handle;
    float *a, *b, *c;
    const float alpha = 1;
    const float beta = 0;

    check(cublasCreate(&handle));

    start = clock();
    check(cudaMallocManaged(&a, N*N * sizeof(float)));
    check(cudaMallocManaged(&b, N*N * sizeof(float)));
    check(cudaMallocManaged(&c, N*N * sizeof(float)));

    for (size_t i = 0; i < N*N; i++) {
        a[i] = i / 37.0;
        b[i] = i / 101.0;
    }
    end = clock();
    log("host: cudaMallocManaged+init", start, end);

    start = clock();
    check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      N, N, N,
                      &alpha,
                      a, N,
                      b, N,
                      &beta,
                      c, N));
    check(cudaDeviceSynchronize());
    end = clock();
    log("cublasSgemm", start, end);

    return 0;
}
