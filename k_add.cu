#include <stdio.h>


__global__ void add_kernel(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main() {
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
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	printf("res: %d\n", c);

    // all are done, so we can free all the memory we have allocated
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}