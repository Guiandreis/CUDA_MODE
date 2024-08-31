
//compute global vector sum C = A +B
// Each thread performs one pair-wise addiction
#include <stdio.h>

__global__ 
void vecAddKernel(float* A, float* B, float*C, int n) {
    // Perform sum value in each tread
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n ) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n) {
    // In device
    // Create device vectors 
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    // Alocate mmemory for device
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // Copy host memory values to device memory
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    // Sum
    vecAddKernel<<<ceil(n/255.0), 255>>>(A_d, B_d, C_d, n);

    // Copy device values to host memory
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%g ", double(C[i]));
    }

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(void) {
    // create host vectors
	float *A_h, *B_h, *C_h;
	int n = 100;
	// allocate memory for host vectors
	A_h = (float*)malloc(sizeof(float)*n);
	B_h = (float*)malloc(sizeof(float)*n);
	C_h = (float*)malloc(sizeof(float)*n);

	for (int i = 0; i < n; i++) {
		A_h[i] = (10);
		B_h[i] = (10);
	}

    vecAdd(A_h, B_h, C_h, n);

    // free host memmory
    free(A_h);
    free(A_h);
    free(C_h);


}