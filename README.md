# <center>**并行计算基础与实践课程项目报告**<center/>



### 一、概述

本报告将扫描问题具体化为求前缀和问题,使用了多种方法实现对扫描算法的并行加速。

主要实现了**CPU串行版本**、**基于GPU的double buffer扫描方法**、**基于GPU的work-efficient扫描方法**，以及**对work-efficient进行避免bank conflict的优化方法**。各方法均可以处理任意规模的数据（不局限于一个Block）。

项目包含以下代码：

![image-20221217160808812](.\imgs\image-20221217160808812.png)

其中，`config.h`记录常量，如BlockSize等；`gpu_timer.h`将计时工具封装起来；`PerformanceTest`对四种方法进行性能测试；`Scan`是扫描算法的基类；`ScanCPU`实现CPU版本的扫描；`ScanDoubleBuffer`实现在GPU上使用double buffer方法的并行扫描；`ScanWorkEfficient`实现在GPU上使用work-efficient方法的并行扫描；`ScanWorkEfficient_BCA`对work-efficient方法的bank conflict问题进行优化。

### 二、Scan

`Scan`是扫描算法的基类。子类需要实现`getPerfixSum`方法。

`timeUsed_w_dataCopy`记录包含设备间数据拷贝的时间，`timeUsed_wo_dataCopy`记录不包含数据拷贝的计算时间。

```c++
// Scan.h
#ifndef __SCAN_H__
#define __SCAN_H__

#include "config.h"
#include <cstdio>

class Scan {
    public:
        Scan(int *a, int len){for(int i=0;i<len;i++)A[i]=a[i];n=len;}
        ~Scan(){};
        int A[MAX_SIZE];
        int prefixSum[MAX_SIZE];
        int n;
        float timeUsed_w_dataCopy;
        float timeUsed_wo_dataCopy;
        virtual void getPrefixSum() = 0;
        void printPrefixSum() const {for(int i=0;i<n;i++)printf("%d ",prefixSum[i]);printf("\n");}
};
#endif
```

### 三、ScanCPU

`ScanCPU`实现CPU版本的扫描，即简单地串行求前缀和，时间复杂度为$O(n)$。

```c++ 
// ScanCPU.h
#include "Scan.h"

class ScanCPU: public Scan {
    public:
        ScanCPU(int *a, int len);
        void getPrefixSum();
};
```

```c++
// ScanCPU.cu
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
```

### 三、ScanDoubleBuffer

`ScanDoubleBuffer`实现在GPU上使用double buffer方法的并行扫描。

朴素的并行算法使用**倍增**的思想，设$s[i][j]$表示 $j-2^i$ 到 $j$ 区间的和，那么$s[i+1][j] = s[i][j-2^i]+s[i][j]$，时间复杂度为$O(nlog_2n)$。如下图所示：

<img src="D:\学习工作\学习资料\大三上\并行计算\作业\大作业\final_project\并行计算课程报告.assets\image-20221217163651647.png" alt="image-20221217163651647" style="zoom: 67%;" />

但只这样不能用于并行求解，因为对一个元素的读写同时在进行。使用**double buffer**方法可以解决这个问题，double buffer类似于滚动数据的思想，每次读和写都在不同的buffer中进行，这样可以实现并行加速。但是只能求解一个Block规模的扫描问题，因为Block之间不能同步。

代码实现如下所示。使用shared memory减少内存带宽的需要，对每个Block开两个BLOCK_SIZE大小的shared memory数组，分别用于一层的读和写。然后模拟倍增算法，最终将shared memory对应位置的值写入`output_prefixSumPerBlock`中，得到此Block的前缀和。

```c++
// ScanDoubleBuffer.cu
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
```

但是只这样不能处理规模大于BLOCK_SIZE的扫描问题。

可以这样做

1. 将长度为N的序列A按BLOCK_SIZE分块，每一个Block内都可以使用`DoubleBuffer`方法求出此Block中的前缀和，得到数组`prefixSumPerBlock`
2. 记录每个Block内元素的和到一个数组B中，再将B求前缀和得到数组`IntermediateSumPerBlock`
3. 最后再将`IntermediateSumPerBlock`的对应值“补偿”到`prefixSumPerBlock`数组中

这样可以求出长度为N的序列的前缀和了。

<img src=".\imgs\image-20221217171702892.png" alt="image-20221217171702892" style="zoom:67%;" />

发现步骤2中”再求前缀和“，其实是一个递归的过程，于是可以用**递归**的思路求解任意长度的序列前缀和了。

代码如下。

```c++
// ScanDoubleBuffer.cu
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
        *d_outdata = d_prefixSum;
        cudaFree(d_prefixSumPerBlock);
        cudaFree(d_intermediateSumPerBlock);
        cudaFree(d_intermediatePrefixSum);
    }
}
```

函数`scan_DoubleBuffer_finalAdd`是求解上述第3步骤的kernel，输入每个Block内部求完前缀和的数组`prefixSumPerBlock`，和每个Block最后一个元素的前缀和`intermediatePrefixSum`，经过简单的相加，输出为最终的所有序列的前缀和`prefixSum`。

函数`getPrefixSumSolution_DoubleBuffer`是求解前缀和的函数，输入为device中的数据`d_A`和长度`length`，输出为指向前缀和输出数组的二级指针。在这个函数中，首先求出`d_A`中每个Block内部的前缀和`d_prefixSumPerBlock`，并将每个Block内部的和放入数组`d_intermediateSumPerBlock`。如果`d_A`中只有一个Block，那么`d_prefixSumPerBlock`就是最终的前缀和了；若有多于1个的Block，则递归调用`getPrefixSumSolution_DoubleBuffer`求解`d_intermediateSumPerBlock`的前缀和，输出为`d_intermediatePrefixSum`，然后再调用kernel `scan_DoubleBuffer_finalAdd`，求解出最终前缀和。

于是，DoubleBuffer方法的`getPrefixSum`函数先将数组A拷贝到device中，然后调用`getPrefixSumSolution_DoubleBuffer`函数求得前缀和，最终从device中拷贝回host。

```c++
// ScanDoubleBuffer.cu
void ScanDoubleBuffer::getPrefixSum() {
    GpuTimer timer1, timer2;
    timer1.Start();
    int *d_A, *d_preSum;
    cudaMalloc((void **)&d_A, n * sizeof(int));
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
```

### 四、ScanWorkEfficient

DoubleBuffer方法的工作效率太低了，使用数据结构中**树状数组**的思想，可以更快地求解扫描问题。这个方法在输入数据上构建一棵二叉树（虚拟的），然后两次遍历这棵树。示意图如下：

<img src=".\imgs\image-20221217180626192.png" alt="image-20221217180626192" style="zoom: 33%;" />

在正向构建这棵树的时候，所有加法都是in place的，如上图这样构建过程时间复杂度是$O(N)$的，构建完成后，第$i$个位置（从0开始编号）存放$a[i-lowbit(i)]$到$a[i]$的和，其中$i-lowbit(i)$表示将i的最低一位1变成0。然后再反向分配，一个节点所有右子树的前缀和都要加上左子树的前缀和，如此反向分配可以求出所有位置的前缀和，反向分配的时间复杂度也是$O(N)$的。

但这样求出来的前缀和是exclusive的，即第一位是0；与DoubleBuffer不同，DoubleBuffer是inclusive的，第一位是$a[1]$。

但是像DoubleBuffer一样只能处理Block以内的前缀和，因为Block之间无法同步。

代码实现如下：

```c++
// ScanWorkEfficient.cu
__global__
void scan_WorkEfficient_OneBlock(int* indata, int* outdata_prefixSumPerBlock, int* outdata_IntermediateSumPerBlock, int length) {
    int thid = threadIdx.x, blid = blockIdx.x;
    __shared__ int shmem[2*BLOCK_SIZE];
    int Block_offset = blid * 2 * BLOCK_SIZE;
    if (Block_offset + 2 * thid < length) shmem[2*thid] = indata[Block_offset+2*thid];
    if (Block_offset + 2 * thid + 1 < length) shmem[2*thid+1] = indata[Block_offset+2*thid+1];
    __syncthreads();
    // 正向构建阶段
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
    //反向分配阶段
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
```

我使用一个线程处理两个元素，所以一个Block可以处理2*BLOCK_SIZE大小的序列。这种方法只能处理长度是2的幂次的序列，但可以通过设置不足2的幂次的剩余元素都为0来解决这个问题。通过规律推导和二叉树性质完成正向构建阶段，反向分配阶段正好相反。最终将Block内部的前缀和写入`outdata_prefixSumPerBlock`中。

与DoubleBuffer一样，这样只能处理Block以内的前缀和，但是可以通过用和DoubleBuffer一样的方法使work-efficient算法可以用于任意长度的序列求前缀和。

代码与DoubleBuffer思路一样，不再多赘述：

```c++
// ScanWorkEfficient.cu

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
        *d_outdata = d_prefixSum;
        cudaFree(d_prefixSumPerBlock);
        cudaFree(d_intermediateSumPerBlock);
        cudaFree(d_intermediatePrefixSum);
    }
}
```



### 五、ScanWorkEfficient_BCA

上面的work-efficient方法会面临严重的bank conflict问题，如正向构建阶段，随着树深度增加，bank conflict先增加后减少，最多的时候能达到$256\div 32=16$ depth的bank conflict。解决方法就是padding，对shared memory第$i$个元素的访问，在padding意义下转变成对第$i+i/32$个元素的访问，这样可以大大减小bank conflict。

代码与上一部分的差别仅在shared memory访问时增加padding。

```c++
// ScanWorkEfficient_BCA.cu
#define CONFLICT_FREE_OFFSET(a) (((a)/32))
__global__
void scan_WorkEfficient_BCA_OneBlock(int* indata, int* outdata_prefixSumPerBlock, int* outdata_IntermediateSumPerBlock, int length) {
    int thid = threadIdx.x, blid = blockIdx.x;
    __shared__ int shmem[2*BLOCK_SIZE+BLOCK_SIZE];
    int Block_offset = blid * 2 * BLOCK_SIZE;
    if (Block_offset + 2 * thid < length) shmem[2*thid+CONFLICT_FREE_OFFSET(2*thid)] = indata[Block_offset+2*thid];
    if (Block_offset + 2 * thid + 1 < length) shmem[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] = indata[Block_offset+2*thid+1];
    __syncthreads();
    int times = 1;
    for (int thread_cnt = BLOCK_SIZE; thread_cnt >= 1; thread_cnt >>= 1, times <<= 1) {
        if (thid < thread_cnt) {
            int b = times * 2 * (thid + 1) - 1;
            int a = b - times;
            shmem[b+CONFLICT_FREE_OFFSET(b)] += shmem[a+CONFLICT_FREE_OFFSET(a)];
        }
        __syncthreads();
    }
    if (thid == 0) shmem[2*BLOCK_SIZE-1+CONFLICT_FREE_OFFSET(2*BLOCK_SIZE-1)] = 0;
    __syncthreads();
    
    for (int thread_cnt = 1; thread_cnt <= BLOCK_SIZE; thread_cnt <<= 1) {
        times >>= 1;
        if (thid < thread_cnt) {
            int b = times * 2 * (thid + 1) - 1;
            int a = b - times;
            // swap(shmem[a], shmem[b]);
            int tmp = shmem[a+CONFLICT_FREE_OFFSET(a)];
            shmem[a+CONFLICT_FREE_OFFSET(a)] = shmem[b+CONFLICT_FREE_OFFSET(b)];
            shmem[b+CONFLICT_FREE_OFFSET(b)] += tmp;
        }
        __syncthreads();
    }
    if (Block_offset + 2 * thid < length) {
        outdata_prefixSumPerBlock[Block_offset+2*thid] = shmem[thid*2+CONFLICT_FREE_OFFSET(thid*2)];
        if (2 * thid == 2 * BLOCK_SIZE - 1) 
            outdata_IntermediateSumPerBlock[blid] = shmem[thid*2+CONFLICT_FREE_OFFSET(thid*2)] + indata[Block_offset+2*thid];
    }
    if (Block_offset + 2 * thid + 1 < length) {
        outdata_prefixSumPerBlock[Block_offset+2*thid+1] = shmem[thid*2+1+CONFLICT_FREE_OFFSET(thid*2+1)];
        if (2 * thid + 1 == 2 * BLOCK_SIZE - 1) 
            outdata_IntermediateSumPerBlock[blid] = shmem[thid*2+1+CONFLICT_FREE_OFFSET(thid*2+1)] + indata[Block_offset+2*thid+1];
    }

}

```



### 六、性能测试

对ScanCPU、ScanDoubleBuffer、ScanWorkEfficient、ScanWorkEfficient_BCA分别执行20次，计算平均用时。

#### 实验环境

* CPU: 12核 Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
* GPU: Tesla P40
* CUDA 11.3

#### 启用O2优化

在**启用O2优化**模式下，输入`make`，执行`./scan_test 123123123`，使用长度为123123123的随机数组进行性能测试。

输出如下：

![启用O2优化实验结果-1](.\imgs\image-20221219212241807.png)

![启用O2优化实验结果-2](.\imgs\image-20221219212252348.png)

每一行第一个时间代表算上设备间数据拷贝的总时间，第二个时间是刨去设备间拷贝的时间。

可以发现GPU并行处理如果刨去设备间数据拷贝的时间，远远快于CPU上串行所用的时间，但是如果加上设备间拷贝的时间，开启O2优化后的CPU串行计算的速度甚至能更快。

而且可以看出WorkEfficient方法确实快过DoubleBuffer方法，并且对WorkEfficient方法进行BCA优化确实有效。

| 方法              | 时间（加上设备间数据拷贝）/ ms | 时间（刨去设备间数据备间拷贝）/ ms |
| ----------------- | ------------------------------ | ---------------------------------- |
| CPU               | ---                            | **131.472**                        |
| DoubleBuffer      | 188.775                        | 30.132                             |
| WorkEfficient     | 180.064                        | 21.528                             |
| WorkEfficient-BCA | 177.612                        | **18.744**                         |

下面观察nvprof给出的详细信息：

![image-20221219212305854](.\imgs\image-20221219212305854.png)

对数据进行处理得到每次计算前缀和过程中各kernel的运行时间：

| 方法              | scan_OneBlock时间/ms | scan_finalAdd时间/ms |
| ----------------- | -------------------- | -------------------- |
| DoubleBuffer      | 15.044               | 4.095                |
| WorkEfficient     | 8.805                | 4.098                |
| WorkEfficient-BCA | **6.166**            | 4.099                |

可以发现`finalAdd`所用时间大致相同，但`scan_OneBlock`时间，WorkEfficient和WorkEfficient-BCA远远小于DoubleBuffer，而WorkEfficient-BCA所用的时间又比WorkEfficient少30%。

#### 关闭O2优化

在**未启用O2优化**的情况下，运行`make`，执行`nvprof .\scan_test 123123123`

结果如下：

#### 各规模数据比较

**关闭O2优化**

`./scan_test 12312312`

![image-20221219220116860](.\imgs\image-20221219220116860.png)

`./scan_test 123123`

![image-20221219220213172](.\imgs\image-20221219220213172.png)

`./scan_test 12312`

![image-20221219220232930](.\imgs\image-20221219220232930.png)

关闭O2优化情况下，GPU所用时间约$N>10^6$时比CPU串行所用时间更短。

**开启O2优化**

`./scan_test 12312312`

![image-20221219220416044](.\imgs\image-20221219220416044.png)

`./scan_test 12312`

![image-20221219220433375](.\imgs\image-20221219220433375.png)

开启O2优化的情况下，若包含设备间数据拷贝，则GPU所用时间将一直大于CPU所用时间。

#### 不同BlockSize的影响

在N=123123123的情况下，观察不同的BlockSize对性能的影响

BLOCKSIZE=1024：

![image-20221219220631720](.\imgs\image-20221219220631720.png)

BLOCKSIZE=512：

![image-20221219220820386](.\imgs\image-20221219220820386.png)

BLOCKSIZE=256：

![image-20221219220911214](.\imgs\image-20221219220911214.png)

BLOCKSIZE=128：

![image-20221219221007794](.\imgs\image-20221219221007794.png)

BLOCKSIZE=64：

![image-20221219221102620](.\imgs\image-20221219221102620.png)

BLOCKSIZE=32：

![image-20221219221148527](.\imgs\image-20221219221148527.png)

当BLOCKSIZE过大时，SM中Block数量有限；当BLOCKSIZE过小时无法充分利用所有线程；当BLOCKSIZE在512时，并行处理性能最高。
