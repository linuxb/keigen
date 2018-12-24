#include <stdio.h>
#include <vector>

__global__ void add_kernel(int *a, int *b, int *c) {
    *c = *a + *b;
}

//template <typename T>
//struct Matrix {
//    int width;
//    int height;
//    std::vector<T> data;
//};
//
//template <>
//struct Matrix {
//    int width;
//    int height;
//    std::vector<float> data;
//};

#define BS 2

typedef struct {
    int width;
	int height;
	int stride;
    float *data;
} Matrix;

// run inner GPU threads
__device__ float GetElement(Matrix mat, int row, int col) {
    return mat.data[row * mat.width + col];
}

__device__ void SetElement(Matrix mat, int row, int col, float val) {
    mat.data[row * mat.stride + col] = val;
}

__device__ Matrix GetSubMatrix(Matrix mat, int row, int col) {
    Matrix subx;
	subx.width = subx.height = BS;
	subx.stride = mat.stride;
	subx.data = &mat.data[row * BS * mat.width + col * BS];
    return subx;
}

// kernel
__global__ void KDot(Matrix A, Matrix B, Matrix C) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    // in RK(inner register in each thread)
    float res = 0;
    // b * b sub matrix
    Matrix csub = GetSubMatrix(C, block_row, block_col);
    int row = threadIdx.y;
    int col = threadIdx.x;
    for (int k = 0; k < A.width / BS; k++) {
        Matrix asub = GetSubMatrix(A, block_row, k);
        Matrix bsub = GetSubMatrix(B, k, block_col);
        // shm set
        __shared__ float a_shm[BS][BS];
        __shared__ float b_shm[BS][BS];
        // each thread load an element from global memory only once
        a_shm[row][col] = GetElement(asub, row, col);
        b_shm[row][col] = GetElement(bsub, row, col);
		__syncthreads();
		// sync all threads ensure all elements in sub matrix are stored to shm
        // compute this op
        for (int e = 0; e < BS; e++) {
            res += a_shm[row][e] * b_shm[e][col];
        }
		__syncthreads();
    }
    SetElement(csub, row, col, res);
}

// call the kernel
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    dim3 dim_block(BS, BS); // block internal
    dim3 dim_grid(A.height / BS, B.width / BS);
    Matrix d_A, d_B, d_C;
    d_A.width = A.width; d_A.height = A.height; d_A.stride = A.stride;
    // only zcopy 'data' field to GPU memory
    size_t size = d_A.width * d_A.height * sizeof(float);
    cudaError_t err = cudaMalloc(&d_A.data, size);
    if (err) {
        printf("Error while allocate matrix A ==> %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    // pointer to dest, pointer to src, size, type
    cudaMemcpy(d_A.data, A.data, size, cudaMemcpyHostToDevice);

    d_B.width = B.width; d_B.height = B.height; d_B.stride = B.stride;
    // only zcopy 'data' field to GPU memory
    size = d_B.width * d_B.height * sizeof(float);
    err = cudaMalloc(&d_B.data, size);
    if (err) {
        printf("Error while allocate matrix B ==> %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    // pointer to dest, pointer to src, size, type
    cudaMemcpy(d_B.data, B.data, size, cudaMemcpyHostToDevice);

    d_C.width = C.width; d_C.height = C.height; d_C.stride = C.stride;
    // only zcopy 'data' field to GPU memory
    size = d_C.width * d_C.height * sizeof(float);
    err = cudaMalloc(&d_C.data, size);
    if (err) {
        printf("Error while allocate matrix C ==> %s\n", cudaGetErrorString(err));
        exit(-1);
    }
	  // invoke kernel
    KDot<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
    err = cudaThreadSynchronize();
    if (err) {
        printf("Error while launch kernel ==> %s\n", cudaGetErrorString(err));
    }
    // zcopy result to host when computation done
    err = cudaMemcpy(C.data, d_C.data, size, cudaMemcpyDeviceToHost);
    if (err) {
        printf("Error while copy result to host memory ==> %s\n", cudaGetErrorString(err));
    }

    // free all these memory finally
    cudaFree(d_A.data);
    cudaFree(d_B.data);
	cudaFree(d_C.data);
}

int main()
{
	Matrix A, B, C;
	// A[6 * 5] B[5 * 6] C[6 * 6]
	A.width = A.stride = B.height = 4;
	A.height = B.stride = B.width = 6;
	C.width = C.height = C.stride = 6;
	// allocate memory for these matrices
	A.data = (float*)malloc(A.width * A.height * sizeof(float));
	B.data = (float*)malloc(B.width * B.height * sizeof(float));
	C.data = (float*)malloc(C.width * C.height * sizeof(float));
	// initialize A B to I[M * N]
	for (int i = 0; i < A.height; i++) {
		for (int j = 0; j < A.width; j++) {
			A.data[i * A.width + j] = 1;
		}
	}
	for (int i = 0; i < B.height; i++) {
		for (int j = 0; j < B.width; j++) {
			B.data[i * B.width + j] = 1;
		}
	}
	MatMul(A, B, C);
	printf("res: \n");
	// print out the result
	for (int i = 0; i < C.height; i++) {
		for (int j = 0; j < C.width; j++) {
			printf("%.1f ", C.data[i * C.width + j]);
		}
		printf("\n");
	}
    return 0;
}
