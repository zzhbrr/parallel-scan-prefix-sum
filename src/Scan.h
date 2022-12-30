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