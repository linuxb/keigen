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

template <int X, int Y>
struct fq {
    // resolve at compile time
    enum {
        type = X + Y
    };
};
template <typename T, int Q> struct kl_tn {};
template <typename T> struct kl_tn<T, 6> { typedef T value; };
void solve() {
    kl_tn<int, fq<3, 3>::type>::value x = 7;
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

template <size_t idx, typename TTuple>
void xfill(TTuple &t) {}

template <size_t idx, typename TTuple, typename TFirst, typename ...TRest>
void xfill(TTuple &t, TFirst &first, TRest ...rest) {
    ::xget<idx>(t) = first;
    xfill<idx + 1>(t, rest...);
}

template <typename ...TTypes>
static KTuple<TTypes...>& MakeTupleImpl(TTypes ...args) {
    KTuple<TTypes...> res;
    xfill<0>(res, args...);
    return res;
}

static void MakeTuple() {
//    KTuple<int, char, std::string> my_tuple;
//    ::xget<0>(my_tuple) = 3;
//    ::xget<1>(my_tuple) = 'k';
//    ::xget<2>(my_tuple) = "lollipop";
    auto my_tuple = MakeTupleImpl(3, 'k', "lollipop");
    std::cout << "tuple elements: \n"
              << YIELDS_FROM_T(0) << "\n"
              << YIELDS_FROM_T(1) << "\n"
              << YIELDS_FROM_T(2) << "\n";
}

// replace type implementation
// replace_type<c, x, y>::type

template <class c, class x, class y>
struct replace_type {
    typedef c type;
};

template <class x, class y>
struct replace_type<x, x, y> {
    typedef y type;
};

template <class x, class y>
struct replace_type<x*, x, y> {
    typedef y* type;
};

template <class x, class y>
struct replace_type<x&, x, y> {
    typedef y& type;
};

template <size_t N, class x, class y>
struct replace_type<x[N], x, y> {
    typedef y type[N];
};

// recursive definition
template <class c, class x, class y>
struct replace_type<c*, x, y> {
    typedef typename replace_type<c, x, y>::type* type;
};

template <class c, class x, class y>
struct replace_type<c&, x, y> {
    typedef typename replace_type<c, x, y>::type& type;
};

template <size_t N, class c, class x, class y>
struct replace_type<c[N], x, y> {
    typedef typename replace_type<c, x, y>::type type[N];
};

// for functor (with or without args)
template <class f, class x, class y>
struct replace_type<f(), x, y> {
    typedef typename replace_type<f, x, y>::type type ();
};

template <class f, class x, class y, class ...Args>
struct replace_type<f(Args...), x, y> {
    typedef typename replace_type<f, x, y>::type
        type (typename replace_type<Args, x, y>::type...);
};

template <int x>
struct ty_int {
    static const int val = x;
};

template <class T1, class T2>
struct fn_add {
    static const int val = T1::val + T2::val;
};

struct functor_add {
    // call other through inheriting
    template <class T1, class T2>
    struct apply : fn_add<T1, T2> {};
};

template <class T>
struct mty {
    using type = T;
};

template <typename E>
struct Tr {
    using type = typename E::type;
};

namespace sfinae {

//    template <class P, class Q = typename mty<int>::type>
//    void pfn(P, Q);
    template <typename T>
    class is_class {
        // is class or struct
        typedef char ktrue[2];
        typedef char kfalse[1];
        // $ty_c get resolved at compile-time, a sfinae error
        // occurs if $ty_c is not a class type
        // declaration only
        template <typename C> static ktrue& check(int C::*);
        // more specific than the first one, so `check(int)` will always be called
        // as I will always love lazyfish~~~
        // more generic (any parameter)
        template <typename C> static kfalse& check(...);

    public:
        static bool const value = sizeof(check<T>(0)) == sizeof(ktrue);
    };

    template <class T>
    class KHas {
        template <class U> static const char check(typename U::type *);
        template <class U> static const int check(...);
        // do member checking
//        template <class U> static const char check(decltype(U::type) *);

    public:
        static const int value = sizeof(check<T>(0)) == sizeof(char);
    };

    template <typename _Ty>
    struct X {
        using type = _Ty;
        type p;
    };
    // U will be set to `void` whenever if there's only one parameter T
    // so the partial specialized one won't be used unless the result
    // of deduction is equal to `void` exactly
    template <class T, class U = void>
    struct xhas_type {
        static const int value = false;
    };
    // partial specialization
//    template <class T>
//    struct xhas_type<T, typename T::type> {
//        static const int value = true;
//    };
    template <typename Any>
    struct Try {
        using type = void;
    };
    // try to resolve T::type, regress toward the generic type when
    // sfinae error occurs
    template <class T>
    struct xhas_type<T, typename Try<typename T::type>::type> {
        static const int value = true;
    };
    // any member detection at compile-time
//    template <class T>
//    struct xhas_type<T, typename Try<decltype(T::val)>::type> {
//        static const int value = true;
//    };
    template <size_t N>
    struct xarg {
        static const size_t value = N;
    };
    namespace placeholders {
        using p_1 = xarg<1>;
        using p_2 = xarg<2>;
    }
    template <size_t N, typename ThisArg, typename ...Rest>
    struct get_arg : get_arg<N - 1, Rest...> {};
    template <typename Arg, typename ...TrueArgs>
    struct lambda_impl {
        using type = Arg;
    };
    template <typename ThisArg, typename ...Rest>
    struct get_arg<1, ThisArg, Rest...> {
        using arg = ThisArg;
    };
    // args mapping
    // partial specialization to handle placeholders
    template <size_t N, typename ...TrueArgs>
    struct lambda_impl<xarg<N>, TrueArgs...> {
        using type = typename get_arg<N, TrueArgs...>::arg;
    };
    template <
            template <typename ...Args> class T,
            typename ...Args,
            typename ...TrueArgs
            >
    struct lambda_impl<T<Args...>, TrueArgs...> {
        using type = typename lambda_impl<T<Args...>>::type::template apply<TrueArgs...>::solved_type;
    };
    template <
            template <typename ...Args> class T,
            typename ...Args
            >
    struct lambda_impl<T<Args...>> {
        // return a meta functor
        struct type {
            template <typename ...TrueArgs>
            struct apply {
                using solved_type = T<typename lambda_impl<Args, TrueArgs...>::type...>;
            };
        };
    };

    template<class T>
    struct lambda {
        using type = typename lambda_impl<T>::type;
    };

    template <class X, class Y>
    struct kmul {
        static const int res = X::res * Y::res;
    };

    template <int n>
    struct k_int {
        static const int res = n;
    };

    int test_lamdba() {
        using p_1 = placeholders::p_1;
        using p_2 = placeholders::p_2;
        using kq1 = lambda<kmul<p_1, p_2>>::type;
        using kqf = lambda<kmul<kmul<p_1, p_1>, kmul<p_2, p_2>>>::type;
        auto res = kqf::apply<k_int<3>, k_int<6>>::solved_type::res;
        return res;
    }

    template <class T>
    class KDtx {
        // use a macro for different names
        struct fallback { int k; };
        // construct controlled ambiguity
        struct Derived : T, fallback {};
        // enforce type checking by compiler
        template <typename U, U> struct Check { using type = U; };
        typedef char ktrue[2];
        typedef char kfalse[1];
        // if `C` do have a member `k`, make sure the type checking successful
        // or an sfinae error also throws away this overload func
        // use decltype() since C++11
        template <class C> static const kfalse& try_(typename Check<int fallback::*, &C::k>::type);
        template <class C> static const ktrue& try_(...);

    public:
        static const int res = sizeof(try_<Derived>(nullptr)) == sizeof(ktrue);
    };
}


static void RunMono() {
    std::cout << "type equation: " << std::is_same<replace_type<int[3][9], int, float>::type,
            float[3][9]>::value << "\n";
    std::cout << "type equation: " << std::is_same<replace_type<int***&(int**, int&, int[3][6]), int, float>::type,
            float***&(float**, float&, float[3][6])>::value << "\n";
    std::cout << "fn_add: " << functor_add::apply<ty_int<3>, ty_int<3>>::val << "\n";
    std::cout << "is_class: " << sfinae::is_class<float>::value << "\n";
    std::cout << "has_mtype: " << sfinae::xhas_type<sfinae::X<std::vector<int>>>::value << "\n";
    std::cout << "has_mtype: " << sfinae::KDtx<sfinae::X<int>>::res << "\n";
    struct Y { int type; };
    std::cout << "has_mtype: " << sfinae::KHas<Y>::value << "\n";
    std::cout << "lambda kq1 res: " << sfinae::test_lamdba() << "\n";
}


int main() {
    run_traits();
    solve();
//    run_loop();
//    run_k();
    MakeTuple();
    RunMono();
    return 0;
}