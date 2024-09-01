#include <stdio.h>
#include <cuda_runtime.h>

int main(){

    int deviceCount = 0;  // Variable to store the number of devices
    cudaError_t err = cudaGetDeviceCount(&deviceCount);  // Get the number of CUDA devices

    if (err != cudaSuccess) {  // Check for errors
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        // return 1;  // Exit if there was an error
    }

    printf("Number of CUDA-capable devices: %d\n", deviceCount);  // Print the number of devices

    // return 0;  // Exit the program

    
    FILE *file = fopen("part_1/chapter_4/cuda_device_count.txt", "w");

    for(unsigned int i = 0; i < deviceCount; i++) {
        cudaDeviceProp devProp;
        err = cudaGetDeviceProperties(&devProp, i);
        if (err != cudaSuccess) {
            // Use cudaGetErrorString to print the error message
            fprintf(stderr, "Failed to get device properties for device %d: %s\n", i, cudaGetErrorString(err));
            continue;  // Continue to the next device if an error occurs
        }
        // Successfully retrieved properties, you can print specific properties here
        fprintf(file, "CUDA Device %d properties successfully retrieved\n", i);
        fprintf(file, "  Device name: %s\n", devProp.name);
        fprintf(file, "  Total global memory: %zu bytes\n", devProp.totalGlobalMem);
        fprintf(file, "\nDevice %d: \"%s\"\n", i, devProp.name);
        fprintf(file, "  Compute capability: %d.%d\n", devProp.major, devProp.minor);
        fprintf(file, "  Total global memory: %zu bytes\n", devProp.totalGlobalMem);
        fprintf(file, "  Shared memory per block: %zu bytes\n", devProp.sharedMemPerBlock);
        fprintf(file, "  Registers per block: %d\n", devProp.regsPerBlock);
        fprintf(file, "  Warp size: %d\n", devProp.warpSize);
        fprintf(file, "  Memory pitch: %zu bytes\n", devProp.memPitch);
        fprintf(file, "  Max threads per block: %d\n", devProp.maxThreadsPerBlock);
        fprintf(file, "  Max threads dimensions (x, y, z): (%d, %d, %d)\n",
            devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        fprintf(file, "  Max grid size (x, y, z): (%d, %d, %d)\n",
            devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        fprintf(file, "  Clock rate: %d kHz\n", devProp.clockRate);
        fprintf(file, "  Total constant memory: %zu bytes\n", devProp.totalConstMem);
        fprintf(file, "  Compute mode: %d\n", devProp.computeMode);
        fprintf(file, "  Concurrent kernels: %s\n", devProp.concurrentKernels ? "Yes" : "No");
        fprintf(file, "  ECC enabled: %s\n", devProp.ECCEnabled ? "Yes" : "No");
        fprintf(file, "  Memory clock rate: %d kHz\n", devProp.memoryClockRate);
        fprintf(file, "  Memory bus width: %d bits\n", devProp.memoryBusWidth);
        fprintf(file, "  L2 cache size: %d bytes\n", devProp.l2CacheSize);
        fprintf(file, "  Maximum texture dimensions (1D): %d\n", devProp.maxTexture1D);
        fprintf(file, "  Maximum texture dimensions (2D): %d x %d\n", devProp.maxTexture2D[0], devProp.maxTexture2D[1]);
        fprintf(file, "  Maximum texture dimensions (3D): %d x %d x %d\n", devProp.maxTexture3D[0], devProp.maxTexture3D[1], devProp.maxTexture3D[2]);
        fprintf(file, "  Multi-processor count: %d\n", devProp.multiProcessorCount);
        fprintf(file, "  Integrated: %s\n", devProp.integrated ? "Yes" : "No");
        fprintf(file, "  Can map host memory: %s\n", devProp.canMapHostMemory ? "Yes" : "No");
        fprintf(file, "  Unified addressing: %s\n", devProp.unifiedAddressing ? "Yes" : "No");
        fprintf(file, "  Cooperative launch: %s\n", devProp.cooperativeLaunch ? "Yes" : "No");
        fprintf(file, "  Cooperative multi-device launch: %s\n", devProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        // Add more fields if needed
        // Add more properties as needed
    }
}