CC=g++ -g -c -Wall -fPIC
CC_SHR=g++ -shared -Wl,-soname
INSTALLDIR=/home/jared/repos/OpenCLPhold/common

all:
	$(CC) clpp.cpp StopWatch.cpp clppContext.cpp clppProgram.cpp clppCount.cpp clppSort.cpp clppSort_CPU.cpp clppSort_RadixSort.cpp clppSort_RadixSortGPU.cpp clppScan_Default.cpp clppScan_GPU.cpp -I../../inc/ -L/usr/local/cuda-7.5/lib64 -lOpenCL
	$(CC_SHR),$(INSTALLDIR)/lib/libclpp.so.1 -o $(INSTALLDIR)/lib/libclpp.so.1.0.1 *.o -lc
	ln -s $(INSTALLDIR)/lib/libclpp.so.1.0.1 $(INSTALLDIR)/lib/libclpp.so.1
	ln -s $(INSTALLDIR)/lib/libclpp.so.1.0.1 $(INSTALLDIR)/lib/libclpp.so
	rm *.o

clean:
	rm $(INSTALLDIR)/lib/libclpp.*
