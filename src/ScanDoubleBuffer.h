#include "Scan.h"

class ScanDoubleBuffer: public Scan {
    public:
        ScanDoubleBuffer(int *a, int len);
        void getPrefixSum();
};