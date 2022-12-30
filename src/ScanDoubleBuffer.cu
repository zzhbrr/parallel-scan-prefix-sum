#include "gpu_timer.h"
#include "ScanDoubleBuffer.h"

ScanDoubleBuffer::ScanDoubleBuffer(int *a, int len): Scan(a, len){}

void getPrefixSumSolution_DoubleBuffer(int* d_A, int **d_outdata, int length);

void ScanDoubleBuffer::getPrefixSum() {
    GpuTimer timer1, timer2;
    timer1.Start();
    int *d_A, *d_preSum;
    cudaMalloc((void **)&d_A, n * sizeof(int));
    // cudaMalloc((void **)&d_preSum, n * sizeof(int));
    cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
    timer2.Start();
    getPrefixSumSolution_DoubleBuffer(d_A, &d_preSum, n);
    timer2.Stop();
    int *ptr_h_preSum = prefixSum;
    cudaMemcpy(ptr_h_preSum, d_preSum, n * sizeof(int), cudaMemcpyDeviceToHost);
    timer1.Stop();
    timeUsed_w_dataCopy = timer1.Elapsed(), timeUsed_wo_dataCopy = timer2.Elapsed();
    cudaFree(d_A);
    cudaFree(d_preSum);
}



__global__
void scan_DoubleBuffer_OneBlock(int* indata, int* outdata_prefixSumPerBlock, int* outdata_IntermediateSumPerBlock, int length) {
	int thid = threadIdx.x, blid = blockIdx.x;
    int id = blid * BLOCK_SIZE + thid;
	int c_read = 0, c_write = 1;
    __shared__ int shmem[2][BLOCK_SIZE];
    if (id < length)
        shmem[c_read][thid] = indata[id];
    __syncthreads();
    for (int offset = 1; offset < length; offset <<= 1) {
        if (thid - offset >= 0) shmem[c_write][thid] = shmem[c_read][thid] + shmem[c_read][thid - offset];
        else shmem[c_write][thid] = shmem[c_read][thid];
        __syncthreads();
        c_write ^= 1, c_read ^= 1;
    }
    if (id < length) {
        outdata_prefixSumPerBlock[id] = shmem[c_read][thid];
        if (thid == BLOCK_SIZE - 1)
            outdata_IntermediateSumPerBlock[blid] = shmem[c_read][thid];
    }
}

__global__
void scan_DoubleBuffer_finalAdd(int *indata_prefixSumPerBlock, int *indata_d_intermediatePrefixSum, int *outdata_prefixSum, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int index = int(i / BLOCK_SIZE);
    if (i < length) {
        if (index != 0) outdata_prefixSum[i] = indata_prefixSumPerBlock[i] + indata_d_intermediatePrefixSum[index-1];
        else outdata_prefixSum[i] = indata_prefixSumPerBlock[i];
    }
}

void getPrefixSumSolution_DoubleBuffer(int* d_A, int **d_outdata, int length) {
    int *d_prefixSumPerBlock;
    int *d_intermediateSumPerBlock;
    cudaMalloc((void **)&d_prefixSumPerBlock, length * sizeof(int));
    int intermediateNum = ceil((double)length/BLOCK_SIZE);
    cudaMalloc((void **)&d_intermediateSumPerBlock, intermediateNum * sizeof(int));

    int GridSize = ceil((double)length/BLOCK_SIZE);
    scan_DoubleBuffer_OneBlock<<<GridSize, BLOCK_SIZE>>>(d_A, d_prefixSumPerBlock, d_intermediateSumPerBlock, length);
    cudaDeviceSynchronize();
    if (intermediateNum == 1) {
        *d_outdata = d_prefixSumPerBlock;
        cudaFree(d_intermediateSumPerBlock);
    } else {
        int *d_intermediatePrefixSum;
        getPrefixSumSolution_DoubleBuffer(d_intermediateSumPerBlock, &d_intermediatePrefixSum, intermediateNum);
        int *d_prefixSum;
        cudaMalloc((void **)&d_prefixSum, length * sizeof(int));
        scan_DoubleBuffer_finalAdd<<<ceil((double)length/BLOCK_SIZE), BLOCK_SIZE>>>(d_prefixSumPerBlock, d_intermediatePrefixSum, d_prefixSum, length);
        cudaDeviceSynchronize();
        // cudaMemcpy(d_outdata, d_prefixSum, length *sizeof(int), cudaMemcpyDeviceToDevice);
        *d_outdata = d_prefixSum;
        cudaFree(d_prefixSumPerBlock);
        cudaFree(d_intermediateSumPerBlock);
        // cudaFree(d_prefixSum);
        cudaFree(d_intermediatePrefixSum);
    }
}