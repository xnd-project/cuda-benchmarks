#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cublasXt.h>

const char *cublasStatusAsString(cublasStatus_t status);
void check(cudaError_t err);
void check(cublasStatus_t status);
void log(const char *prefix, clock_t start, clock_t end);


#endif
