//
// Created by nuxeslin on 2018/9/13.
//
#include <iostream>

using namespace std;

int main() {
    auto rref = std::move(6);
    cout << "res: " << rref << endl;
    return 0;
}
