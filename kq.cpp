//
// Created by nuxeslin on 2018/11/7.
//
#include <iostream>

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

struct Matrix {
    int width;
    int height;
    float* data;
};

int main() {
    size_t ss = 0;
    auto rref = std::move(3);
    float mat[6][6] = {1};
    char my_key[] = "lollipop";
    memset(my_key, 'k', 6);
    printf("%s\n", my_key);
    InitMat<6, 6>(mat, 6, 6);
    Matrix cmat = {6, 5};
    cmat.data = (float*)malloc(cmat.width * cmat.height * sizeof(float));
    cmat.data[3] = 1;
    printf("%.1f", cmat.data[3]);
    return 0;
}