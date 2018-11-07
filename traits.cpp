#include <iostream>
#include <vector>

using namespace std;

template <class T>
class KL_iter {
public:
    typedef T value_type;
    T *ptr;
    explicit KL_iter(T *p = 0) : ptr(p) {};
    T& operator* () const { return *ptr; }
};

template <class T>
struct kl_traits {
    typedef typename T::value_type value_type;
};

template <class T>
struct kl_traits<T*> {
    typedef T value_type;
};

template <class I>
typename kl_traits<I>::value_type iter_get(I iter) { return *iter; }


void pfn(int arg) {
    std::cout << "type of arg (int)" << std::endl;
}

void pfn(char arg) {
    std::cout << "type of arg (char)" << std::endl;
}

void pfn(double arg) {
    std::cout << "type of arg (double)" << std::endl;
}

void run_traits() {
    kl_traits<vector<int>>::value_type a;
    pfn(a);
    kl_traits<vector<char>>::value_type b;
    pfn(b);
    kl_traits<vector<double>>::value_type c;
    pfn(c);
    kl_traits<int *>::value_type x;
    pfn(x);
    KL_iter<int> my_iter(new int(6));
    const char y = 'y';
    cout << "res: " << iter_get(&y) << endl;
    cout << "res: " << iter_get(my_iter) << endl;
}

int main() {
    run_traits();
    return 0;
}