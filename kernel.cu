#include <stdio.h>
#include <vector>

__global__ add_kernel(int *a, int *b, int *c) {
    *c = *a + *b;
}

template <typename T>
struct Matrix {
    int width;
    int height;
    std::vector<T> data;
};

__global__ void kDot(Matrix A, Matrix B, Matrix C) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    float res = 0;
    // b * b sub matrix
    Matrix csub = GetSubMatrix(C, block_row, block_col);
    int row = threadIdx.y;
    int col = threadIdx.x;
    for (int k = 0; k < A.width; k++) {
        asub = GetSubMatrix(A, k, block_row);
        bsub = GetSubMatrix(B, block_col, k);
        // shm set
        __shared__ float a_shm[bs][bs];
        __shared__ float b_shm[bs][bs];
        // each thread load an element from global memory only once
        a_shm[row][col] = GetElement(asub, row, col);
        b_shm[row][col] = GetElement(bsub, row, col);
        __syncthreads();
        // compute this op
        for (int e = 0; e < bs; e++) {
            bs += asub[row][e] * bsub[e][col];
        }
        __syncthreads();
    }
    SetElement(csub, row, col, res);
}

int main()
{
    // on Host
    int a, b, c;
    // copy on Device
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);
    // allocate memory on device
    //use a pointer to address to be populated
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);


    // initialize a, b
    a = 4;
    b = 2;
    // copy memory from host to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // luanch the kernel to compute
    add_kernel<<<1, 1>>>(d_a, d_b, d_c);

    // since the result we need still be stored at device
    // so we have to copy it to host memory
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeciveToHost);

    // all are done, so we can free all the memory we have allocated
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
