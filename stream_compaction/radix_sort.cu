#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix_sort.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer() {
            static PerformanceTimer t;
            return t;
        }

        __global__ void uniformAdd(int n,
            int* odata,
            const int num) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] += num;
        }

       __global__ void negate_bools_into(int n, int* out, const int* in) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            out[index] = 1 - in[index];
        }

       __global__ void radix_to_bools(int n, int* b, const int* idata, int bit) {
           int index = (blockIdx.x * blockDim.x) + threadIdx.x;
           if (index >= n) {
               return;
           }
           unsigned int u = static_cast<unsigned int>(idata[index]);
           b[index] = (u >> bit) & 1;
       }

        __global__ void assign_indexes(int n,
            int* odata, const int* idata, const int* isOne, const int* idxOnes, const int* idxZeros, int totalZeroes) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            int one = isOne[index];
            if (one) {
                odata[idxOnes[index] + totalZeroes] = idata[index];
            }
            else {
                odata[idxZeros[index]] = idata[index];
            }
        }

        void onestep(int n, int m, int* odata, const int* idata,
            int* idxOnes, int* idxZeros, int* bOne, int* bZero, int bit) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // To bools
            RadixSort::radix_to_bools << <fullBlocksPerGrid, blockSize >> > (m, bOne, idata, bit);
            cudaDeviceSynchronize();
            if (n > m) {
                cudaMemset(bOne + m, 0, (n - m) * sizeof(int));
            }
            cudaDeviceSynchronize();

            RadixSort::negate_bools_into <<<fullBlocksPerGrid, blockSize >> > (m, bZero, bOne);
            cudaDeviceSynchronize();
            if (n > m) {
                cudaMemset(bZero + m, 0, (n - m) * sizeof(int));
            }
            cudaDeviceSynchronize();
            
            // Scan ones -> idxOnes (exclusive)
            Efficient::recursiveScan(n, idxOnes, bOne);
            cudaDeviceSynchronize();

            int onesBeforeLast = 0, lastOneBit = 0;
            cudaMemcpy(&onesBeforeLast, idxOnes + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy onesBeforeLast failed!");
            cudaMemcpy(&lastOneBit, bOne + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy lastOneBit failed!");
            int totalOnes = onesBeforeLast + lastOneBit;
            int totalZeros = m - totalOnes;  // use m so padding never contributes
            // Scan zeros -> idxZeros (exclusive)
            Efficient::recursiveScan(n, idxZeros, bZero);
            cudaDeviceSynchronize();
            // Scatter: use original isOne flags
            RadixSort::assign_indexes << <fullBlocksPerGrid, blockSize >> > (m, odata, idata, bOne, idxOnes, idxZeros, totalZeros);
            checkCUDAErrorFn("Assign indexes failed!");
            cudaDeviceSynchronize();
        }

        void sort(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }
            if (n == 1) {          // handle trivial case without GPU work
                odata[0] = idata[0];
                return;
            }
            int m = n;
            int pow2 = 1 << ilog2ceil(n);
            if (n != pow2) {
                n = pow2;
            }
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int* dev_bufA = nullptr;
            int* dev_bufB = nullptr;
            int* t = nullptr;
            int* f = nullptr;
            int* b0 = nullptr;
            int* b1 = nullptr;

            cudaMalloc((void**)&dev_bufA, m * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_buf_A failed!");

            cudaMalloc((void**)&dev_bufB, m * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_buf_B failed!");

            cudaMalloc((void**)&t, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc t failed!");

            cudaMalloc((void**)&f, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc f failed!");

            cudaMalloc((void**)&b0, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc b failed!");
            cudaMalloc((void**)&b1, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc b failed!");

            cudaMemcpy(dev_bufA, idata, m * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("MemCpy dev_buf_A failed!");

            int num_iter = sizeof(int) * 8;

            // ------------------GPU-----------------------------------
            timer().startGpuTimer();
            for (int bit = 0; bit < num_iter; bit++) {
                RadixSort::onestep(n, m, dev_bufB, dev_bufA, t, f, b1, b0, bit);
                checkCUDAErrorFn("onestep Radix failed!");
                cudaDeviceSynchronize();
                std::swap(dev_bufA, dev_bufB);  // Output now in dev_buf_A
            }
            timer().endGpuTimer();
            // ------------------GPU-----------------------------------

            cudaMemcpy(odata, dev_bufA, m * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cudaFree(dev_bufA);
            checkCUDAErrorFn("CudaFree dev_buf_A failed!");
            cudaFree(dev_bufB);
            checkCUDAErrorFn("CudaFree dev_buf_B failed!");
            cudaFree(t);
            checkCUDAErrorFn("CudaFree t failed!");
            cudaFree(f);
            checkCUDAErrorFn("CudaFree f failed!");
            cudaFree(b0);
            checkCUDAErrorFn("CudaFree b0 failed!");
            cudaFree(b1);
            checkCUDAErrorFn("CudaFree b1 failed!");
        }
    }
}
