#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cublasXt.h>

const char *cublasStatusAsString(cublasStatus_t status);
void check(cudaError_t err);
void check(cublasStatus_t status);
size_t checked_strtosize(const char *v);
size_t checked_mul(size_t a, size_t b);
void log(const char *prefix, clock_t start, clock_t end);


#endif
