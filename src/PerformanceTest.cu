#include <cstdio>
#include <cstdlib>
#include "config.h"
#include "ScanCPU.h"
#include "ScanWorkEfficient.h"
#include "ScanDoubleBuffer.h"
#include "ScanWorkEfficient_BCA.h"
int A[MAX_SIZE];
int N;
bool verbose;

bool check_correctness(int *gt, int *b, int n, int typ) {
    // typ = 0, check with DoubleBuffer
    if (typ == 0) {
        for (int i = 0; i < n; i ++) {
            if (gt[i] != b[i]) {
                printf("check error: position %d with gt[%d]=%d, b[%d]=%d\n", i, i, gt[i], i, b[i]);
                return false;
            }
        }
        return true;
    }
    // typ = 1, check with WorkEfficient
    if (typ == 1) {
        for (int i = 0; i < n - 1; i ++) {
            if (gt[i] != b[i+1]) {
                printf("check error: position %d with gt[%d]=%d, b[%d]=%d\n", i, i, gt[i], i, b[i]);
                return false;
            }
        }
        return true;
    }
    return true;
}

__global__
void deviceWarmup_vectorCopy(int *A, int *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    B[i] = A[i];
}

void deviceWarmup() {
    int *d_A;
    cudaMalloc((void **)&d_A, N * sizeof(int));
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    deviceWarmup_vectorCopy<<<128, 128>>>(d_A, d_A);
    cudaFree(d_A);
}

void runScan(Scan* scan, int times) {
    float timeUsed_w_dataCopy = 0, timeUsed_wo_dataCopy = 0;
    for (int i = 1; i <= times; i ++) {
        scan->getPrefixSum();
        timeUsed_w_dataCopy += scan->timeUsed_w_dataCopy;
        timeUsed_wo_dataCopy += scan->timeUsed_wo_dataCopy;
    }
    if (verbose) scan->printPrefixSum();
    printf("%.3fms, %.3fms\n", timeUsed_w_dataCopy/times, timeUsed_wo_dataCopy/times);
}

ScanCPU *scan_cpu;
ScanDoubleBuffer *scan_DoubleBuffer;
ScanWorkEfficient *scan_WorkEfficient;
ScanWorkEfficient_BCA *scan_WorkEfficient_BCA;


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("not enough args");
        return 0;
    }
    N = atoi(argv[1]);
    verbose = false;
    if (argc >= 3) {
        if (atoi(argv[2]) == 0) verbose = false;
        if (atoi(argv[2]) == 1) verbose = true;
    }
    srand(time(0));
    for (int i = 0; i < N; i ++) A[i] = rand()%10;
    printf("N = %d, BLOCK_SIZE = %d\n", N, BLOCK_SIZE);
    if (verbose) {
        printf("A: ");
        for (int i = 0; i < N; i ++) printf("%d ", A[i]);printf("\n");
    }
    
    deviceWarmup();

    printf("scan_cpu: ");
    scan_cpu = new ScanCPU(A, N);
    runScan(scan_cpu, 20);

    printf("scan_DoubleBuffer: ");
    scan_DoubleBuffer = new ScanDoubleBuffer(A, N);
    runScan(scan_DoubleBuffer, 20);
    
    printf("scan_WorkEfficient: ");
    scan_WorkEfficient = new ScanWorkEfficient(A, N);
    runScan(scan_WorkEfficient, 20);

    printf("scan_WorkEfficient_BCA: ");
    scan_WorkEfficient_BCA = new ScanWorkEfficient_BCA(A, N);
    runScan(scan_WorkEfficient_BCA, 20);

    if (check_correctness(scan_cpu->prefixSum, scan_DoubleBuffer->prefixSum, N, 0)) {
        printf("check [double buffer] answer correct!\n");
    }
    if (check_correctness(scan_cpu->prefixSum, scan_WorkEfficient->prefixSum, N, 1)) {
        printf("check [work efficient] answer correct!\n");
    }
    if (check_correctness(scan_cpu->prefixSum, scan_WorkEfficient_BCA->prefixSum, N, 1)) {
        printf("check [work efficient bca] answer correct!\n");
    }
    return 0;
}