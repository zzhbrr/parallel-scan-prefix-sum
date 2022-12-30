#include "Scan.h"

class ScanWorkEfficient: public Scan {
    public:
        ScanWorkEfficient(int *a, int len);
        void getPrefixSum();
};
