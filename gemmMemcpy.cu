#include <stdlib.h>
#include <stdio.h>
#include <cublasXt.h>
#include <cuda_runtime.h>
#include "common.hh"


int main(int argc, char *argv[])
{
    size_t N = DEFAULT_N;
    if (argc==2) N = (size_t)atoi(argv[1]);
    printf("N=%zd\n", N);

    clock_t start_program, end_program;
    clock_t start, end;
    cublasHandle_t handle;
    float *a, *b, *c;
    float *da, *db, *dc;
    const float alpha = 1;
    const float beta = 0;
    const size_t count = N*N * sizeof(float);

    start_program = clock();

    check(cublasCreate(&handle));

    start = clock();
    check(cudaMallocHost(&a, count));
    check(cudaMallocHost(&b, count));
    check(cudaMallocHost(&c, count));

    for (size_t i = 0; i < N*N; i++) {
        a[i] = i / 37.0;
        b[i] = i / 101.0;
    }
    end = clock();
    log("host: cudaMallocHost+init", start, end);

    start = clock();
    check(cudaMalloc(&da, count));
    check(cudaMalloc(&db, count));
    check(cudaMalloc(&dc, count));

    check(cudaMemcpy(da, a, count, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b, count, cudaMemcpyHostToDevice));

    check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      N, N, N,
                      &alpha,
                      a, N,
                      b, N,
                      &beta,
                      c, N));

    check(cudaMemcpy(c, dc, count, cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    end = clock();
    log("cublasSgemm", start, end);

    start = clock();
    for (size_t i = 0; i < N; i++) {
        if (a[i] < 0 || b[i] < 0 || c[i] < 0) {
            fprintf(stderr, "unexpected result a: %f  b: %f  c: %f\n",
                    a[i], b[i], c[i]);
            exit(1);
        }
    }
    end = clock();
    log("host: access all arrays", start, end);

    start = clock();
    for (size_t i = 0; i < N; i++) {
        if (a[i] < 0 || b[i] < 0 || c[i] < 0) {
            fprintf(stderr, "unexpected result a: %f  b: %f  c: %f\n",
                    a[i], b[i], c[i]);
            exit(1);
        }
    }
    end = clock();
    log("host: access all arrays a second time", start, end);

    start = clock();
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    end = clock();
    log("host: free", start, end);

    end_program = clock();
    log("total", start_program, end_program);

    return 0;
}
