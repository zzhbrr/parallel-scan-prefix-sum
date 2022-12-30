NVCC        = nvcc
NVCC_FLAGS  =  --ptxas-options=-v 
NVCC_FLAGS  += -O2
#LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64

EXE			= scan_test
OBJ			= ScanCPU.o ScanDoubleBuffer.o ScanWorkEfficient.o PerformanceTest.o ScanWorkEfficient_BCA.o

default: $(EXE)

ScanCPU.o: ScanCPU.cu
	$(NVCC) -c -o $@ ScanCPU.cu $(NVCC_FLAGS)

ScanDoubleBuffer.o: ScanDoubleBuffer.cu
	$(NVCC) -c -o $@ ScanDoubleBuffer.cu $(NVCC_FLAGS)

ScanWorkEfficient.o: ScanWorkEfficient.cu
	$(NVCC) -c -o $@ ScanWorkEfficient.cu $(NVCC_FLAGS)

ScanWorkEfficient_BCA.o: ScanWorkEfficient_BCA.cu
	$(NVCC) -c -o $@ ScanWorkEfficient_BCA.cu $(NVCC_FLAGS)

PerformanceTest.o: PerformanceTest.cu
	$(NVCC) -c -o $@ PerformanceTest.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)

rmlog:
	rm -rf *.txt
