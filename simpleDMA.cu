#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cuda_runtime.h>
#include "common.hh"


static __global__ void
f(const uint64_t a[], const uint64_t b[], uint64_t c[], int64_t N)
{
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = index; i < N; i += stride) {
        c[i] = a[i] * b[i];
    }
}

static void
doit(const uint64_t a[], const uint64_t b[], uint64_t c[], int64_t N)
{
    int blockSize = 256;
    int64_t numBlocks = (N + blockSize - 1) / blockSize;

    f<<<numBlocks, blockSize>>>(a, b, c, N);
}

int
main(int argc, char *argv[])
{
    size_t N = 10000000;
    clock_t start_program, end_program;
    clock_t start, end;
    uint64_t *a, *b, *c;
    size_t count;

    if (argc == 2) {
        N = checked_strtosize(argv[1]);
    }
    count = checked_mul(N, sizeof(uint64_t));

    /* Initialize context */
    check(cudaMallocHost(&a, 128));
    check(cudaDeviceSynchronize());
    check(cudaFreeHost(a));

    start_program = clock();

    start = clock();
    check(cudaMallocHost(&a, count));
    check(cudaMallocHost(&b, count));
    check(cudaMallocHost(&c, count));
    end = clock();
    log("host: MallocHost", start, end);

    start = clock();
    for (size_t i = 0; i < N; i++) {
        a[i] = 3;
        b[i] = 5;
    }
    end = clock();
    log("host: init arrays", start, end);

    start = clock();
    doit(a, b, c, N);
    check(cudaDeviceSynchronize());
    end = clock();
    log("device: DMA+compute+synchronize", start, end);

    start = clock();
    for (size_t i = 0; i < N; i++) {
        if (a[i] != 3 || b[i] != 5 || c[i] != 15) {
            fprintf(stderr, "unexpected result a: %lu  b: %lu  c: %lu\n",
                    a[i], b[i], c[i]);
            exit(1);
        }
    }
    end = clock();
    log("host: access all arrays", start, end);

    start = clock();
    for (size_t i = 0; i < N; i++) {
        if (a[i] != 3 || b[i] != 5 || c[i] != 15) {
            fprintf(stderr, "unexpected result a: %lu  b: %lu  c: %lu\n",
                    a[i], b[i], c[i]);
            exit(1);
        }
    }
    end = clock();
    log("host: access all arrays a second time", start, end);

    start = clock();
    check(cudaFreeHost(a));
    check(cudaFreeHost(b));
    check(cudaFreeHost(c));
    end = clock();
    log("host: free", start, end);

    end_program = clock();
    log("total", start_program, end_program);

    return 0;
}
