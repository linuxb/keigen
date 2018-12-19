//
// Created by nuxeslin on 2018/11/7.
//
#include <iostream>
#include <Eigen/Core>
#include <stdlib.h>

using namespace std;
using namespace Eigen;

template <int M, int N>
void InitMat(float mat[M][N], int row_num, int col_num) {
//    for (int i = 0; i < row_num; i++) {
//        for (int j = 0; j < col_num; j++) {
//            mat[i * col_num + j] = 1;
//        }
//    }
    for (int k = 0; k < row_num * col_num; k++) {
        *((float*)mat + k) = 1;
    }
}

template <size_t N>
struct MyAllocator {
    char buf[N];
    void *ptr;
    size_t sz;
    MyAllocator() : ptr(buf), sz(N) {}
    template <typename T>
    T* aligned_malloc(size_t a = alignof(T)) {
        if (std::align(a, sizeof(T), ptr, sz)) {
            auto res = reinterpret_cast<T*>(ptr);
            ptr = (char*)res + sizeof(T);
            sz -= sizeof(T);
            return res;
        }
        printf("aligned_malloc error!\n");
        return nullptr;
    }
};

struct KMatrix {
    int width;
    int height;
    float* data;
};

// set memory alignment
// 16 bytes alignment in osx
// #pragma pack parameter to be '1', '2', '4', '8', or '16', other number makes no sense, compiler
// will use the default value 16
#pragma pack(16)

struct Area {
    char x1;
//    char x2[16];
    double x2;
    // max size is 8 bytes
    double x3;
};

void Run() {
    int size = 3;
    Eigen::VectorXf v(size), u(size), w(size);
    u = v + w;
    std::cout << u << std::endl;
}

// at runtime
void KL_bad_allocate() {
    auto huge = static_cast<size_t>(-1);
    ::operator new(huge);
}

void KL_compute_op() {
    Eigen::Matrix2d mat;
    KL_bad_allocate();
//    std::bad_alloc();
}

void AlignedAllocate() {
    char n = 'x';
    float q = 2.0;
    printf("y1: 0x%lx\n", (size_t)&n);
    // without customized alignment
    // alignment equal to min(sizeof(float), 16) = 4 bytes
    // so there will be 3 bytes padding subsequently
    // - (4 + 3 = 7) bytes
    printf("y2: 0x%lx\n", (size_t)&q);
    // alignment at heap
    printf("at heap:\n");
//    auto pool = (Area*)malloc(sizeof(Area) * 3);
    Area *pool;
    // guarantee the address of `pool` multiple of 32(bytes), while `malloc` will not
    if (posix_memalign(reinterpret_cast<void**>(&pool), 32, 64 * 3)) {
        printf("error while allocate aligned memory\n");
    }
//    pool = (Area*)malloc(24 * 3);
    // if we don't access to these memory, the calculation below only operator
    // the pointer, which add some offset to the base address of `pool`, then it
    // will be casted to `char*`
    // when we access a slot with am index larger than the capacity(3) using `pool[1000]`
    // nothing happens, but segment fault occurs when write some data to that slot
    // but after `malloc` called, nothing happens again, in spite of `pool[1000]`
    // which is overflow absolutely, so it just compute the position which will be filled,
    // UNSAFE memory operation !!! without any boundary checking
    pool[0] = {'k', 6};
    pool[1] = {'o', 4};
    // it works
//    pool[1000] = {'e', 7, 2};
    printf("r0: %ld\n", ((size_t)&pool[0]) % 32);
    printf("p1: 0x%lx\n", (size_t)&pool[1]);
    // + 48 bytes
    printf("p2: 0x%lx\n", (size_t)&pool[2]);
    printf("offset: %lu\n", ((char*)&pool[1] - (char*)&pool[0]));
//    printf("%c\n", pool[1].x1);
}

void Zmalloc() {
    AlignedAllocate();
    // allocated at stack, grows downward
    // but in the case which allocates an array (consecutive area)
    // the stack pointer moves down (-N bytes, N is the length of this array), and its
    // final position is the lowest address in allocated space
    // so we can access an element by *(buffer + i), let the pointer move up
    char buffer[6] = "kkkkk";
    // initialize to 0 when allocated at heap
    auto ptr = (float*)malloc(4);
    // non-successive at heap
    // arithmetic on a pointer to void is invalid
    auto other = (float*)malloc(4);
//    posix_memalign((void**)&ptr, 16, 4);
    cout << "ptr: " << (void*)ptr << endl;
    printf("other: 0x%lx\n", (size_t)other);
    // only pointers to compatible type can be subtracted
    printf("ptr offset: %lu\n", (other - ptr));
    printf("alignof T: %lu\n", alignof(char));
    // their position in memory layout aren't successive (malloc at heap not at stack)
    MyAllocator<128> allocator;
    cout << "ptr0: " << allocator.ptr << endl;
    auto ptr1 = allocator.aligned_malloc<char>();
    *ptr1 = 'x';
    cout << "ptr1: " << (void*)ptr1 << endl;
    // internal pointer of allocator will be adjusted when std::align has been invoked
    // aligned at runtime
    auto ptr2 = allocator.aligned_malloc<int>(8);
    *ptr2 = 7;
    cout << "ptr2: " << (void*)ptr2 << endl;
    // size is 48 bytes when alignment is 16 bytes, and 32 bytes when 8 bytes, 28 bytes when 4 bytes
    printf("size of area: %lu\n", sizeof(Area));
    Area ka = {'x', 5};
    // object of type size_t can safely store a pointer (non-member)
    // can not use static_cast here, but '(size_t)' works
    auto safe_ptr = reinterpret_cast<size_t>(&(ka));
    // must use unsigned long here, or the ptr of size_t(8 bytes) will downcast into object of 4 bytes int type
    // so it just show the lowest 4 bytes
    // use %lx or %lu (8 bytes) when print a pointer or something of type size_t
    printf("safe ptr: %lx\n", safe_ptr);
    printf("sizeof safeptr: %lu\n", sizeof(safe_ptr));
//    printf("%s\n", ka.x2);
    free(ptr);
    free(other);
}

int main() {
    size_t ss = 0;
    auto rref = std::move(3);
    float mat[6][6] = {1};
    char my_key[] = "lollipop";
    memset(my_key, 'k', 6);
    printf("%s\n", my_key);
    InitMat<6, 6>(mat, 6, 6);
    ::KMatrix cmat = {6, 5};
    cmat.data = (float*)malloc(cmat.width * cmat.height * sizeof(float));
    cmat.data[3] = 1;
    printf("%.1f\n", cmat.data[3]);
    Run();
    Zmalloc();
    return 0;
}