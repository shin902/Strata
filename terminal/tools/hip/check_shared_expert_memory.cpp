#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cassert>
#include <cstdio>

__global__ void read_and_update(int* values) {
    const int i = threadIdx.x;
    values[i] = values[i] * 3 + 7;
}

int main() {
    hipDeviceProp_t properties{};
    assert(hipGetDeviceProperties(&properties, 0) == hipSuccess);
    assert(properties.integrated);
    int* host = nullptr;
    int* device = nullptr;
    assert(hipHostMalloc(reinterpret_cast<void**>(&host), 4096,
                         hipHostMallocMapped | hipHostMallocNonCoherent) == hipSuccess);
    assert(hipHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0) == hipSuccess);
    for (int i = 0; i < 256; ++i) host[i] = i;
    assert(hipDeviceSynchronize() == hipSuccess);
    hipLaunchKernelGGL(read_and_update, dim3(1), dim3(256), 0, nullptr, device);
    assert(hipGetLastError() == hipSuccess);
    assert(hipDeviceSynchronize() == hipSuccess);
    for (int i = 0; i < 256; ++i) assert(host[i] == i * 3 + 7);
    assert(hipHostFree(host) == hipSuccess);
    std::puts("CPU/GPU read/write parity passed on one mapped physical allocation");
    assert(hipExtMallocWithFlags(reinterpret_cast<void**>(&device), 4096, hipDeviceMallocFinegrained) == hipSuccess);
    for (int i = 0; i < 256; ++i) device[i] = i;
    assert(hipDeviceSynchronize() == hipSuccess);
    hipLaunchKernelGGL(read_and_update, dim3(1), dim3(256), 0, nullptr, device);
    assert(hipGetLastError() == hipSuccess);
    assert(hipDeviceSynchronize() == hipSuccess);
    for (int i = 0; i < 256; ++i) assert(device[i] == i * 3 + 7);
    assert(hipFree(device) == hipSuccess);
    std::puts("CPU/GPU read/write parity passed on one fine-grained device allocation");
}
