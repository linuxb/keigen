#include <iostream>
#include <vector>
#include <string>

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

template <int X, int Y>
struct lambdaq {
    // resolve at compile time
    enum {
        type = X + Y
    };
};
template <typename T, int Q> struct kl_tn {};
template <typename T> struct kl_tn<T, 6> { typedef T value; };
void solve() {
    kl_tn<int, lambdaq<3, 3>::type>::value x = 7;
    std::cout << "inferred value: " << x << "\n";
}
template <class I>
typename kl_traits<I>::value_type iter_get(I iter) { return *iter; }

template <int n> struct fn {
    static const auto value = fn<n - 1>::value + fn<n - 2>::value;
};
template <> struct fn<1> { static const auto value = 1; };
template <> struct fn<0> { static const auto value = 0; };

static void run_loop() {
    std::cout << "res of loop: " << fn<8>::value << "\n";
}

void pfn(int arg) {
    std::cout << "type of arg (int)" << std::endl;
}

void pfn(char arg) {
    std::cout << "type of arg (char)" << std::endl;
}

void pfn(double arg) {
    std::cout << "type of arg (double)" << std::endl;
}

#ifndef KL_ZAMLLOC
#define ZMALLOC 0
#endif

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

static void run_k() {
    auto start =  std::chrono::steady_clock::now();
    for(size_t i = 0; i < 1000000; i++) {
        // cost 43.72s using recursive function
//        fib(20);
        // only cost 0.001s!!!
        auto res = fn<20>::value;
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "cost time: " << elapsed.count() << "\n";
}

template <typename ...TTypes> class KTuple;
// set up the base case first (empty tuple type)
template <> class KTuple<> {};
// set up the successor via inheriting from KTuple<...TRest> recursively
// which get resolved at compile-time
// so that each $type can be casted into its super class using `static_cast`
template <typename TValue, typename ...TRest>
class KTuple<TValue, TRest...> : public KTuple<TRest...> {
public:
    TValue value;
};
// set up a helper class which yields the corresponding type of the element at
// a given index
// set up the base case (index 0) first which will return the correct element type
// and the variant KTuple class (which stores the value we want) respectively
// Basically, one variant class (contains a TValue) stores a value
// all of them form a inheriting chain as a tuple at runtime
template <size_t idx, typename TTuple> struct KTupleElement;
template <typename TValue, typename ...TRest>
struct KTupleElement<0, KTuple<TValue, TRest...>> {
    using type_t = TValue;
    using KTupleType_t = KTuple<TValue, TRest...>;
};
// establish the inheriting chain
// Backtracking towards 0, each time moves the position 1 step
template <size_t idx, typename TValue, typename ...TRest>
struct KTupleElement<idx, KTuple<TValue, TRest...>> : public KTupleElement<idx - 1, KTuple<TRest...>> { /* empty */ };
// Getter & setter
// Compiler deduces all template args through args in `get`
template <size_t idx, typename ...TTypes>
typename KTupleElement<idx, KTuple<TTypes...>>::type_t& xget(KTuple<TTypes...>& tuple) {
    using KTupleType_t = typename KTupleElement<idx, KTuple<TTypes...>>::KTupleType_t;
    return static_cast<KTupleType_t&>(tuple).value;
};

#define YIELDS_FROM_T(idx) \
    ::xget<idx>(my_tuple)

static void MakeTuple() {
    KTuple<int, char, std::string> my_tuple;
    ::xget<0>(my_tuple) = 3;
    ::xget<1>(my_tuple) = 'k';
    ::xget<2>(my_tuple) = "lollipop";
    std::cout << "tuple elements: \n"
              << YIELDS_FROM_T(0) << "\n"
              << YIELDS_FROM_T(1) << "\n"
              << YIELDS_FROM_T(2) << "\n";
}

int main() {
    run_traits();
    solve();
    run_loop();
    run_k();
    MakeTuple();
    return 0;
}