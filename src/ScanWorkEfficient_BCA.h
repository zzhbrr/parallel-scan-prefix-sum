#include "Scan.h"

class ScanWorkEfficient_BCA: public Scan {
    public:
        ScanWorkEfficient_BCA(int *a, int len);
        void getPrefixSum();
};
