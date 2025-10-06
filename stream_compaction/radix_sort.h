#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        // Reuse the same timer interface as other modules
        StreamCompaction::Common::PerformanceTimer& timer();

        // Stable LSD radix sort
        // Sorts idata into odata. Length is n.
        void sort(int n, int* odata, const int* idata);
    }
}

