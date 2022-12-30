#include "gpu_timer.h"
#include "ScanWorkEfficient.h"

void getPrefixSumSolution_WorkEfficient(int* d_A, int **d_outdata, int length);

ScanWorkEfficient::ScanWorkEfficient(int *a, int len): Scan(a, len){}
void ScanWorkEfficient::getPrefixSum() {
    GpuTimer timer1, timer2;
    timer1.Start();
    int *d_A, *d_preSum;
    cudaMalloc((void **)&d_A, n * sizeof(int));
    // cudaMalloc((void **)&d_preSum, n * sizeof(int));
    cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
    timer2.Start();
    getPrefixSumSolution_WorkEfficient(d_A, &d_preSum, n);
    timer2.Stop();
    int *ptr_h_preSum = prefixSum;
    cudaMemcpy(ptr_h_preSum, d_preSum, n * sizeof(int), cudaMemcpyDeviceToHost);
    timer1.Stop();
    timeUsed_w_dataCopy = timer1.Elapsed(), timeUsed_wo_dataCopy = timer2.Elapsed();
    cudaFree(d_A);
    cudaFree(d_preSum);
}

__global__
void scan_WorkEfficient_OneBlock(int* indata, int* outdata_prefixSumPerBlock, int* outdata_IntermediateSumPerBlock, int length) {
    int thid = threadIdx.x, blid = blockIdx.x;
    __shared__ int shmem[2*BLOCK_SIZE];
    int Block_offset = blid * 2 * BLOCK_SIZE;
    if (Block_offset + 2 * thid < length) shmem[2*thid] = indata[Block_offset+2*thid];
    if (Block_offset + 2 * thid + 1 < length) shmem[2*thid+1] = indata[Block_offset+2*thid+1];
    __syncthreads();
    int times = 1;
    for (int thread_cnt = BLOCK_SIZE; thread_cnt >= 1; thread_cnt >>= 1, times <<= 1) {
        if (thid < thread_cnt) {
            int b = times * 2 * (thid + 1) - 1;
            int a = b - times;
            shmem[b] += shmem[a];
        }
        __syncthreads();
    }
    if (thid == 0) shmem[2*BLOCK_SIZE-1] = 0;
    __syncthreads();
    
    for (int thread_cnt = 1; thread_cnt <= BLOCK_SIZE; thread_cnt <<= 1) {
        times >>= 1;
        if (thid < thread_cnt) {
            int b = times * 2 * (thid + 1) - 1;
            int a = b - times;
            // swap(shmem[a], shmem[b]);
            int tmp = shmem[a];
            shmem[a] = shmem[b];
            shmem[b] += tmp;
        }
        __syncthreads();
    }
    if (Block_offset + 2 * thid < length) {
        outdata_prefixSumPerBlock[Block_offset+2*thid] = shmem[thid*2];
        if (2 * thid == 2 * BLOCK_SIZE - 1) 
            outdata_IntermediateSumPerBlock[blid] = shmem[thid*2] + indata[Block_offset+2*thid];
    }
    if (Block_offset + 2 * thid + 1 < length) {
        outdata_prefixSumPerBlock[Block_offset+2*thid+1] = shmem[thid*2+1];
        if (2 * thid + 1 == 2 * BLOCK_SIZE - 1) 
            outdata_IntermediateSumPerBlock[blid] = shmem[thid*2+1] + indata[Block_offset+2*thid+1];
    }
}

__global__
void scan_WorkEfficient_finalAdd(int *indata_prefixSumPerBlock, int *indata_d_intermediatePrefixSum, int *outdata_prefixSum, int length) {
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    int j = i + 1;
    int index = int(i / (BLOCK_SIZE*2));
    if (i < length) {
        outdata_prefixSum[i] = indata_prefixSumPerBlock[i] + indata_d_intermediatePrefixSum[index];
    }
    index = int(j / (BLOCK_SIZE*2));
    if (j < length) {
        outdata_prefixSum[j] = indata_prefixSumPerBlock[j] + indata_d_intermediatePrefixSum[index];
    }
}


void getPrefixSumSolution_WorkEfficient(int* d_A, int **d_outdata, int length) {
    int *d_prefixSumPerBlock;
    int *d_intermediateSumPerBlock;
    cudaMalloc((void **)&d_prefixSumPerBlock, length * sizeof(int));
    int intermediateNum = ceil((double)length/(BLOCK_SIZE*2));
    cudaMalloc((void **)&d_intermediateSumPerBlock, intermediateNum * sizeof(int));

    int GridSize = ceil((double)length/(BLOCK_SIZE*2));
    scan_WorkEfficient_OneBlock<<<GridSize, BLOCK_SIZE>>>(d_A, d_prefixSumPerBlock, d_intermediateSumPerBlock, length);
    cudaDeviceSynchronize();
    if (intermediateNum == 1) {
        *d_outdata = d_prefixSumPerBlock;
        cudaFree(d_intermediateSumPerBlock);
    } else {
        int *d_intermediatePrefixSum;
        getPrefixSumSolution_WorkEfficient(d_intermediateSumPerBlock, &d_intermediatePrefixSum, intermediateNum);
        int *d_prefixSum;
        cudaMalloc((void **)&d_prefixSum, length * sizeof(int));
        scan_WorkEfficient_finalAdd<<<ceil((double)length/(BLOCK_SIZE*2)), BLOCK_SIZE>>>(d_prefixSumPerBlock, d_intermediatePrefixSum, d_prefixSum, length);
        cudaDeviceSynchronize();
        // cudaMemcpy(d_outdata, d_prefixSum, length *sizeof(int), cudaMemcpyDeviceToDevice);
        *d_outdata = d_prefixSum;
        cudaFree(d_prefixSumPerBlock);
        cudaFree(d_intermediateSumPerBlock);
        // cudaFree(d_prefixSum);
        cudaFree(d_intermediatePrefixSum);
    }
}
