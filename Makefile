

NVCC = nvcc --compiler-options="-Wall -Wextra -O3" -std=c++11 -arch=compute_61 -code=sm_61 -lcublas


default: common.o simpleMemcpy simpleManaged gemmMemcpy gemmManaged gemmXtOutOfCore gemmManagedOutOfCore


simpleMemcpy: Makefile simpleMemcpy.cu common.o
	$(NVCC) -o simpleMemcpy simpleMemcpy.cu common.o -DDEFAULT_N=1000000

simpleManaged: Makefile simpleManaged.cu common.o
	$(NVCC) -o simpleManaged simpleManaged.cu common.o -DDEFAULT_N=1000000

gemmMemcpy: Makefile gemmMemcpy.cu common.o
	$(NVCC) -o gemmMemcpy gemmMemcpy.cu common.o -DDEFAULT_N=1000

gemmManaged: Makefile gemmManaged.cu common.o
	$(NVCC) -o gemmManaged gemmManaged.cu common.o -DDEFAULT_N=1000

gemmXtOutOfCore: Makefile gemmXtOutOfCore.cu common.o
	$(NVCC) -o gemmXtOutOfCore gemmXtOutOfCore.cu common.o -DDEFAULT_N=2000

gemmManagedOutOfCore: Makefile gemmManagedOutOfCore.cu common.o
	$(NVCC) -o gemmManagedOutOfCore gemmManagedOutOfCore.cu common.o -DDEFAULT_N=2000

common.o: Makefile common.cc common.hh
	$(NVCC) -c common.cc


clean:
	rm -f common.o simpleMemcpy simpleManaged gemmMemcpy gemmManaged gemmXtOutOfCore gemmManagedOutOfCore
