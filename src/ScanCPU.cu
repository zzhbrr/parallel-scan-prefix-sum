#include "gpu_timer.h"
#include "ScanCPU.h"

ScanCPU::ScanCPU(int *a, int len): Scan(a, len){}
void ScanCPU::getPrefixSum() {
    GpuTimer timer1, timer2;
    timer1.Start(), timer2.Start();
    prefixSum[0] = A[0];
    for (int i = 1; i < n; i ++) 
        prefixSum[i] = prefixSum[i - 1] + A[i];
    timer2.Stop();
    timer1.Stop();
    timeUsed_w_dataCopy = timer1.Elapsed(), timeUsed_wo_dataCopy = timer2.Elapsed();
}