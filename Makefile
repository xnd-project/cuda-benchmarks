

NVCC = nvcc --compiler-options="-Wall -Wextra -O3" -std=c++11 -arch=compute_61 -code=sm_61 -lcublas


default: common.o simpleMemcpy simpleManaged simpleManagedPrefetch simpleDMA stridedManaged gemmMemcpy gemmManaged gemmManagedPrefetch gemmXtOutOfCore gemmManagedOutOfCore


simpleMemcpy: Makefile simpleMemcpy.cu common.o
	$(NVCC) -o simpleMemcpy simpleMemcpy.cu common.o

simpleManaged: Makefile simpleManaged.cu common.o
	$(NVCC) -o simpleManaged simpleManaged.cu common.o

simpleManagedPrefetch: Makefile simpleManagedPrefetch.cu common.o
	$(NVCC) -o simpleManagedPrefetch simpleManagedPrefetch.cu common.o

simpleDMA: Makefile simpleDMA.cu common.o
	$(NVCC) -o simpleDMA simpleDMA.cu common.o

stridedManaged: Makefile stridedManaged.cu common.o
	$(NVCC) -o stridedManaged stridedManaged.cu common.o

gemmMemcpy: Makefile gemmMemcpy.cu common.o
	$(NVCC) -o gemmMemcpy gemmMemcpy.cu common.o

gemmManaged: Makefile gemmManaged.cu common.o
	$(NVCC) -o gemmManaged gemmManaged.cu common.o

gemmManagedPrefetch: Makefile gemmManagedPrefetch.cu common.o
	$(NVCC) -o gemmManagedPrefetch gemmManagedPrefetch.cu common.o

gemmXtOutOfCore: Makefile gemmXtOutOfCore.cu common.o
	$(NVCC) -o gemmXtOutOfCore gemmXtOutOfCore.cu common.o

gemmManagedOutOfCore: Makefile gemmManagedOutOfCore.cu common.o
	$(NVCC) -o gemmManagedOutOfCore gemmManagedOutOfCore.cu common.o

common.o: Makefile common.cc common.hh
	$(NVCC) -c common.cc


clean:
	rm -f common.o simpleMemcpy simpleManaged simpleManagedPrefetch simpleDMA stridedManaged gemmMemcpy gemmManaged gemmManagedPrefetch gemmXtOutOfCore gemmManagedOutOfCore
