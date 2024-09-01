#include <stdio.h>


// Define the size of the matrices (e.g., 16x16)
#define N 16

// CUDA kernel for matrix multiplication
__global__ void MatrixMulKernel(float *d_A, float *d_B, float *d_C, int width) {
    // Calculate the row index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the element
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        // Perform the dot product of the row and column
        for (int i = 0; i < width; i++) {
            sum += d_A[row * width + i] * d_B[i * width + col];
        }
        // Write the result to the output matrix
        d_C[row * width + col] = sum;
    }
}

// Function to initialize matrices with random values
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = 10;
    }
}

// Function to print the matrix
void printMatrix(const float *matrix, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%0.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main(void) {
    // Host matrices
    float h_A[N * N], h_B[N * N], h_C[N * N];

    // Initialize matrices with random values
    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);

    // Device matrices
    float *d_A, *d_B, *d_C;
    size_t bytes = N * N * sizeof(float);

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

   // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

   // Define the block size and grid size
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    // Print the input and output matrices
    printf("Matrix A:\n");
    printMatrix(h_A, N);
    printf("\nMatrix B:\n");
    printMatrix(h_B, N);
    printf("\nMatrix C (Result):\n");
    printMatrix(h_C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
