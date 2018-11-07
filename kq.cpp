//
// Created by nuxeslin on 2018/11/7.
//
#include <iostream>

void init_mat(float *mat, int row_num, int col_num) {
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            mat[i * col_num + j] = 1.0;
        }
    }
}

int main() {
    auto rref = std::move(6);
    float mat[6][6];
    return 0;
}