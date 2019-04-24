#include <iostream>

//
// Created by nuxeslin on 2019/4/12.
//
class AsyncBase {
    int _ids;

    ~X() = default;
};


class Task : public AsyncBase {
    char *ptr;

    ~Y() {
        delete ptr;
    }
};


static void run() {

}


int main() {
    auto rref = std::move(6);
    std::cout << rref << "\n";
    return 0;
}





