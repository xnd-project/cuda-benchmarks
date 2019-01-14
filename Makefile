

NVCC = nvcc --compiler-options="-Wall -Wextra -O3" -std=c++11 -arch=compute_61 -code=sm_61 -lcublas


default: common.o simpleMemcpy simpleManaged simpleDMA gemmMemcpy gemmManaged gemmXtOutOfCore gemmManagedOutOfCore


simpleMemcpy: Makefile simpleMemcpy.cu common.o
	$(NVCC) -o simpleMemcpy simpleMemcpy.cu common.o

simpleManaged: Makefile simpleManaged.cu common.o
	$(NVCC) -o simpleManaged simpleManaged.cu common.o

simpleDMA: Makefile simpleDMA.cu common.o
	$(NVCC) -o simpleDMA simpleDMA.cu common.o

gemmMemcpy: Makefile gemmMemcpy.cu common.o
	$(NVCC) -o gemmMemcpy gemmMemcpy.cu common.o

gemmManaged: Makefile gemmManaged.cu common.o
	$(NVCC) -o gemmManaged gemmManaged.cu common.o

gemmXtOutOfCore: Makefile gemmXtOutOfCore.cu common.o
	$(NVCC) -o gemmXtOutOfCore gemmXtOutOfCore.cu common.o

gemmManagedOutOfCore: Makefile gemmManagedOutOfCore.cu common.o
	$(NVCC) -o gemmManagedOutOfCore gemmManagedOutOfCore.cu common.o

common.o: Makefile common.cc common.hh
	$(NVCC) -c common.cc


clean:
	rm -f common.o simpleMemcpy simpleManaged simpleDMA gemmMemcpy gemmManaged gemmXtOutOfCore gemmManagedOutOfCore
