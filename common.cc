#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublasXt.h>
#include "common.hh"

const char *
cublasStatusAsString(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
        return "unknown cublas status";
    }
}

void
check(cudaError_t err)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorName(err));
        exit(1);
    }
}

void
check(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s\n", cublasStatusAsString(status));
        exit(1);
    }
}

void
log(const char *prefix, clock_t start, clock_t end)
{
    printf("%s: %f\n", prefix, (double)(end-start)/(double)CLOCKS_PER_SEC);
}
