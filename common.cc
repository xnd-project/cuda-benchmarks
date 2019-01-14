#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <cerrno>
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

size_t
checked_strtosize(const char *v)
{
    char *endptr;
    long long lld;

    errno = 0;
    lld = strtoll(v, &endptr, 10);
    if (*v == '\0' || *endptr != '\0') {
        fprintf(stderr, "N: invalid integer: '%s'\n", v);
        exit(1);
    }

    if (errno == ERANGE || lld < 1 || (uint64_t)lld > SIZE_MAX) {
        fprintf(stderr, "N: out of range: '%s'\n", v);
        exit(1);
    }

    return (size_t)lld;
}

size_t
checked_mul(size_t a, size_t b)
{
    if (a > SIZE_MAX / b) {
        fprintf(stderr, "overflow error\n");
        exit(1);
    }

    return a * b;
}

void
log(const char *prefix, clock_t start, clock_t end)
{
    printf("%s: %f\n", prefix, (double)(end-start)/(double)CLOCKS_PER_SEC);
}
