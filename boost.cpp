//
// Created by nuxeslin on 2018/9/13.
//
#include <iostream>

using namespace std;

namespace kq {
    template <typename TW>
    struct wrapper {
        template <typename T, typename Dummy = void>
        struct fn {
            constexpr static size_t value = 0;
        };
        // explicit specialization of `fn` in non-explicit
        // specialized class scope `wrapper` is invalid
        // we can use a trivial arg, to turn it into a partial specialization
        template <typename Dummy>
        struct fn<int> {
            constexpr static size_t value = 1;
        };
    };

    struct stat {
        template <bool Cond>
        class kcheck {
            template <bool C, typename std::enable_if<C>::type* = nullptr>
            static char try_(void*);

            template <bool C, typename std::enable_if<!C>::type* = nullptr>
            static int try_(void*);

        public:
            constexpr static bool res = sizeof(try_<Cond>(0)) == sizeof(char);
        };
    };
}

static void RunMono() {
//    auto r = kq::wrapper<char>::template fn<int>::value;
    std::cerr << "cond_res: " << kq::stat::template kcheck<false>::res << "\n";
}

int main() {
    auto rref = std::move(6);
    cerr << "res: " << rref << endl;
    RunMono();
    return 0;
}
