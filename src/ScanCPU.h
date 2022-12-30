#include "Scan.h"

class ScanCPU: public Scan {
    public:
        ScanCPU(int *a, int len);
        void getPrefixSum();
};