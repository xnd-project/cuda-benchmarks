#include <cstdio>
#include <cinttypes>
#include <cuda_runtime.h>
#include "common.hh"


static __global__ void
f(const uint64_t x0[], const uint64_t x1[], uint64_t x2[],
  const int64_t s0, const int64_t s1, const int64_t s2,
  int64_t N)
{
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = index; i < N; i += stride) {
        const int64_t i0 = i * s0;
        const int64_t i1 = i * s1;
        const int64_t i2 = i * s2;
        x2[i2] = x0[i0] * x1[i1];
    }
}

static void
doit(const uint64_t a0[], const uint64_t a1[], uint64_t a2[],
     const int64_t s0, const int64_t s1, const int64_t s2,
     int64_t N)
{
    int blockSize = 256;
    int64_t numBlocks = (N + blockSize - 1) / blockSize;

    f<<<numBlocks, blockSize>>>(a0, a1, a2, s0, s1, s2, N);
}

int
main(int argc, char *argv[])
{
    size_t N = 1000000;
    clock_t start_program, end_program;
    clock_t start, end;
    uint64_t *x0, *x1, *x2;
    size_t count;
    const int64_t s0 = 37;
    const int64_t s1 = 101;
    const int64_t s2 = 311;
    size_t i, k0, k1, k2;

    if (argc == 2) {
        N = checked_strtosize(argv[1]);
    }
    count = checked_mul(N, sizeof(uint64_t));

    /* Initialize context */
    check(cudaMallocManaged(&x0, 128));
    check(cudaDeviceSynchronize());
    check(cudaFree(x0));

    start_program = clock();

    start = clock();
    check(cudaMallocManaged(&x0, count*s0));
    check(cudaMallocManaged(&x1, count*s1));
    check(cudaMallocManaged(&x2, count*s2));
    end = clock();
    log("host: MallocManaged", start, end);

    for (size_t i = 0; i < N*s0; i++) {
        x0[i] = UINT64_MAX;
    }
    for (size_t i = 0; i < N*s1; i++) {
        x1[i] = UINT64_MAX;
    }

    start = clock();
    for (i=0, k0=0, k1=0; i<N; i++, k0+=s0, k1+=s1) {
        x0[k0] = 3;
        x1[k1] = 5;
    }
    end = clock();
    log("host: init arrays", start, end);

    start = clock();
    doit(x0, x1, x2, s0, s1, s2, N);
    check(cudaDeviceSynchronize());
    end = clock();
    log("device: uvm+compute+synchronize", start, end);

    start = clock();
    for (i=0, k0=0, k1=0, k2=0; i<N; i++, k0+=s0, k1+=s1, k2+=s2) {
        if (x0[k0] != 3 || x1[k1] != 5 || x2[k2] != 15) {
            fprintf(stderr, "unexpected result x0: %lu  x1: %lu  x2: %lu\n",
                    x0[k0], x1[k1], x2[k2]);
            exit(1);
        }
    }
    end = clock();
    log("host: access all arrays", start, end);

    start = clock();
    for (i=0, k0=0, k1=0, k2=0; i<N; i++, k0+=s0, k1+=s1, k2+=s2) {
        if (x0[k0] != 3 || x1[k1] != 5 || x2[k2] != 15) {
            fprintf(stderr, "unexpected result x0: %lu  x1: %lu  x2: %lu\n",
                    x0[k0], x1[k1], x2[k2]);
            exit(1);
        }
    }
    end = clock();
    log("host: access all arrays a second time", start, end);

    start = clock();
    check(cudaFree(x0));
    check(cudaFree(x1));
    check(cudaFree(x2));
    end = clock();
    log("host: free", start, end);

    end_program = clock();
    log("total", start_program, end_program);

    return 0;
}
