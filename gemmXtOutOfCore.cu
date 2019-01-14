#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublasXt.h>
#include "common.hh"


const size_t N = 32000;


int main()
{
    clock_t start, end; 
    cublasXtHandle_t handle;
    int devices[1] = {0};
    float *a, *b, *c;
    const float alpha = 1;
    const float beta = 0;

    check(cublasXtCreate(&handle));
    check(cublasXtDeviceSelect(handle, 1, devices));

    start = clock();
    check(cudaMallocHost(&a, N*N * sizeof(float)));
    check(cudaMallocHost(&b, N*N * sizeof(float)));
    check(cudaMallocHost(&c, N*N * sizeof(float)));

    for (size_t i = 0; i < N*N; i++) {
        a[i] = i / 37.0;
        b[i] = i / 101.0;
    }
    end = clock();
    log("host: cudaMallocHost+init", start, end);

    start = clock();
    check(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N,
                        &alpha,
                        a, N,
                        b, N,
                        &beta,
                        c, N));
    end = clock();
    log("cublasXtSgemm", start, end);

    return 0;
}
