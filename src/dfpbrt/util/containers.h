#ifndef DFPBRT_UTIL_CONTAINERS_H
#define DFPBRT_UTIL_CONTAINERS_H
#include <dfpbrt/dfpbrt.h>

namespace dfpbrt{
    // TypePack Definition
template <typename... Ts>
struct TypePack {
    static constexpr size_t count = sizeof...(Ts);
};
// TypePack Operations
template <typename T, typename... Ts>
struct IndexOf {
    static constexpr int count = 0;
    static_assert(!std::is_same_v<T, T>, "Type not present in TypePack");
};

template <typename T, typename... Ts>
struct IndexOf<T, TypePack<T, Ts...>> {
    static constexpr int count = 0;
};

template <typename T, typename U, typename... Ts>
struct IndexOf<T, TypePack<U, Ts...>> {
    static constexpr int count = 1 + IndexOf<T, TypePack<Ts...>>::count;
};

}


#endif