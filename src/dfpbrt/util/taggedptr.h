#ifndef DFPBRT_UTIL_TAGGEDPTR_H
#define DFPBRT_UTIL_TAGGEDPTR_H

#include <dfpbrt/util/check.h>
#include <dfpbrt/util/containers.h>

namespace dfpbrt{

    namespace detail {

// TaggedPointer Helper Templates

/**
 * Dispatch<Function type, Return type, Total_avilable_type types...>(Funtion_type && f, ptr, typeindex)
 * for example, TaggedPointer<T1, T2, T3>, then a Dispatch func might be like Dispatch<F, R, T1, T2, T3>(F &&func, void *ptr, int index), and index is in [0, 2]
 */
template <typename F, typename R, typename T>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_EQ(0, index);//for count==1, index is always 0
    return func((const T *)ptr);
}

template <typename F, typename R, typename T>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_EQ(0, index);
    return func((T *)ptr);
}

template <typename F, typename R, typename T0, typename T1>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);//for count ==2, index can be 0 or 1
    DCHECK_LT(index, 2);

    if (index == 0)
        return func((const T0 *)ptr);
    else
        return func((const T1 *)ptr);
}

template <typename F, typename R, typename T0, typename T1>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 2);

    if (index == 0)
        return func((T0 *)ptr);
    else
        return func((T1 *)ptr);
}

template <typename F, typename R, typename T0, typename T1, typename T2>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 3);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    default:
        return func((const T2 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 3);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    default:
        return func((T2 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 4);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    default:
        return func((const T3 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 4);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    default:
        return func((T3 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 5);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    default:
        return func((const T4 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 5);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    default:
        return func((T4 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 6);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    case 4:
        return func((const T4 *)ptr);
    default:
        return func((const T5 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 6);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    case 4:
        return func((T4 *)ptr);
    default:
        return func((T5 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 7);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    case 4:
        return func((const T4 *)ptr);
    case 5:
        return func((const T5 *)ptr);
    default:
        return func((const T6 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 7);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    case 4:
        return func((T4 *)ptr);
    case 5:
        return func((T5 *)ptr);
    default:
        return func((T6 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 8);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    case 4:
        return func((const T4 *)ptr);
    case 5:
        return func((const T5 *)ptr);
    case 6:
        return func((const T6 *)ptr);
    default:
        return func((const T7 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 8);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    case 4:
        return func((T4 *)ptr);
    case 5:
        return func((T5 *)ptr);
    case 6:
        return func((T6 *)ptr);
    default:
        return func((T7 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7, typename... Ts,
          typename = typename std::enable_if_t<(sizeof...(Ts) > 0)>>
DFPBRT_CPU_GPU R Dispatch(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    case 4:
        return func((const T4 *)ptr);
    case 5:
        return func((const T5 *)ptr);
    case 6:
        return func((const T6 *)ptr);
    case 7:
        return func((const T7 *)ptr);
    default:
        return Dispatch<F, R, Ts...>(func, ptr, index - 8);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7, typename... Ts,
          typename = typename std::enable_if_t<(sizeof...(Ts) > 0)>>
DFPBRT_CPU_GPU R Dispatch(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    case 4:
        return func((T4 *)ptr);
    case 5:
        return func((T5 *)ptr);
    case 6:
        return func((T6 *)ptr);
    case 7:
        return func((T7 *)ptr);
    default:
        return Dispatch<F, R, Ts...>(func, ptr, index - 8);
    }
}

template <typename F, typename R, typename T>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_EQ(0, index);
    return func((const T *)ptr);
}

template <typename F, typename R, typename T>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_EQ(0, index);
    return func((T *)ptr);
}

template <typename F, typename R, typename T0, typename T1>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 2);

    if (index == 0)
        return func((const T0 *)ptr);
    else
        return func((const T1 *)ptr);
}

template <typename F, typename R, typename T0, typename T1>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 2);

    if (index == 0)
        return func((T0 *)ptr);
    else
        return func((T1 *)ptr);
}

template <typename F, typename R, typename T0, typename T1, typename T2>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 3);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    default:
        return func((const T2 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 3);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    default:
        return func((T2 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 4);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    default:
        return func((const T3 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 4);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    default:
        return func((T3 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 5);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    default:
        return func((const T4 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 5);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    default:
        return func((T4 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 6);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    case 4:
        return func((const T4 *)ptr);
    default:
        return func((const T5 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 6);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    case 4:
        return func((T4 *)ptr);
    default:
        return func((T5 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 7);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    case 4:
        return func((const T4 *)ptr);
    case 5:
        return func((const T5 *)ptr);
    default:
        return func((const T6 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 7);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    case 4:
        return func((T4 *)ptr);
    case 5:
        return func((T5 *)ptr);
    default:
        return func((T6 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 8);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    case 4:
        return func((const T4 *)ptr);
    case 5:
        return func((const T5 *)ptr);
    case 6:
        return func((const T6 *)ptr);
    default:
        return func((const T7 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, 8);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    case 4:
        return func((T4 *)ptr);
    case 5:
        return func((T5 *)ptr);
    case 6:
        return func((T6 *)ptr);
    default:
        return func((T7 *)ptr);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7, typename... Ts,
          typename = typename std::enable_if_t<(sizeof...(Ts) > 0)>>
auto DispatchCPU(F &&func, const void *ptr, int index) {
    DCHECK_GE(index, 0);

    switch (index) {
    case 0:
        return func((const T0 *)ptr);
    case 1:
        return func((const T1 *)ptr);
    case 2:
        return func((const T2 *)ptr);
    case 3:
        return func((const T3 *)ptr);
    case 4:
        return func((const T4 *)ptr);
    case 5:
        return func((const T5 *)ptr);
    case 6:
        return func((const T6 *)ptr);
    case 7:
        return func((const T7 *)ptr);
    default:
        return DispatchCPU<F, R, Ts...>(func, ptr, index - 8);
    }
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7, typename... Ts,
          typename = typename std::enable_if_t<(sizeof...(Ts) > 0)>>
auto DispatchCPU(F &&func, void *ptr, int index) {
    DCHECK_GE(index, 0);

    switch (index) {
    case 0:
        return func((T0 *)ptr);
    case 1:
        return func((T1 *)ptr);
    case 2:
        return func((T2 *)ptr);
    case 3:
        return func((T3 *)ptr);
    case 4:
        return func((T4 *)ptr);
    case 5:
        return func((T5 *)ptr);
    case 6:
        return func((T6 *)ptr);
    case 7:
        return func((T7 *)ptr);
    default:
        return DispatchCPU<F, R, Ts...>(func, ptr, index - 8);
    }
}

template <typename... Ts>
struct IsSameType;
template <>
struct IsSameType<> {
    static constexpr bool value = true;
};
template <typename T>
struct IsSameType<T> {
    static constexpr bool value = true;
};

template <typename T, typename U, typename... Ts>
struct IsSameType<T, U, Ts...> {
    static constexpr bool value = (std::is_same_v<T, U> && IsSameType<U, Ts...>::value);
};

template <typename... Ts>
struct SameType;
template <typename T, typename... Ts>
struct SameType<T, Ts...> {
    using type = T;
    static_assert(IsSameType<T, Ts...>::value, "Not all types in pack are the same");
};

template <typename F, typename... Ts>
struct ReturnType {
    using type = typename SameType<typename std::invoke_result_t<F, Ts *>...>::type;
};

template <typename F, typename... Ts>
struct ReturnTypeConst {
    using type = typename SameType<typename std::invoke_result_t<F, const Ts *>...>::type;
};

}  // namespace detail


// TaggedPointer Definition https://pbr-book.org/4ed/Utilities/Containers_and_Memory_Management#sec:tagged-pointer
template <typename... Ts>
class TaggedPointer {
  public:
    // TaggedPointer Public Types
    using Types = TypePack<Ts...>;

    // TaggedPointer Public Methods
    TaggedPointer() = default;
    template <typename T>
    DFPBRT_CPU_GPU TaggedPointer(T *ptr) {
        uint64_t iptr = reinterpret_cast<uint64_t>(ptr);
        DCHECK_EQ(iptr & ptrMask, iptr);
        constexpr unsigned int type = TypeIndex<T>();
        bits = iptr | ((uint64_t)type << tagShift);
    }

    DFPBRT_CPU_GPU
    TaggedPointer(std::nullptr_t np) {}

    DFPBRT_CPU_GPU
    TaggedPointer(const TaggedPointer &t) { bits = t.bits; }
    DFPBRT_CPU_GPU
    TaggedPointer &operator=(const TaggedPointer &t) {
        bits = t.bits;
        return *this;
    }

    template <typename T>
    DFPBRT_CPU_GPU static constexpr unsigned int TypeIndex() {
        using Tp = typename std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Tp, std::nullptr_t>)
            return 0;
        else
            return 1 + dfpbrt::IndexOf<Tp, Types>::count;
    }

    DFPBRT_CPU_GPU
    unsigned int Tag() const { return ((bits & tagMask) >> tagShift); }
    template <typename T>
    DFPBRT_CPU_GPU bool Is() const {
        return Tag() == TypeIndex<T>();
    }

    DFPBRT_CPU_GPU
    static constexpr unsigned int MaxTag() { return sizeof...(Ts); }
    DFPBRT_CPU_GPU
    static constexpr unsigned int NumTags() { return MaxTag() + 1; }

    DFPBRT_CPU_GPU
    explicit operator bool() const { return (bits & ptrMask) != 0; }

    DFPBRT_CPU_GPU
    bool operator<(const TaggedPointer &tp) const { return bits < tp.bits; }

    template <typename T>
    DFPBRT_CPU_GPU T *Cast() {
        DCHECK(Is<T>());
        return reinterpret_cast<T *>(ptr());
    }

    template <typename T>
    DFPBRT_CPU_GPU const T *Cast() const {
        DCHECK(Is<T>());
        return reinterpret_cast<const T *>(ptr());
    }

    template <typename T>
    DFPBRT_CPU_GPU T *CastOrNullptr() {
        if (Is<T>())
            return reinterpret_cast<T *>(ptr());
        else
            return nullptr;
    }

    template <typename T>
    DFPBRT_CPU_GPU const T *CastOrNullptr() const {
        if (Is<T>())
            return reinterpret_cast<const T *>(ptr());
        else
            return nullptr;
    }

    std::string ToString() const {
        return std::format("[ TaggedPointer ptr: 0x{0} tag: {1} ]", ptr(), Tag());
    }

    DFPBRT_CPU_GPU
    bool operator==(const TaggedPointer &tp) const { return bits == tp.bits; }
    DFPBRT_CPU_GPU
    bool operator!=(const TaggedPointer &tp) const { return bits != tp.bits; }

    DFPBRT_CPU_GPU
    void *ptr() { return reinterpret_cast<void *>(bits & ptrMask); }

    DFPBRT_CPU_GPU
    const void *ptr() const { return reinterpret_cast<const void *>(bits & ptrMask); }

    // dispatch the function mission
    template <typename F>
    DFPBRT_CPU_GPU decltype(auto) Dispatch(F &&func) {
        DCHECK(ptr());//Check the ptr is not a nullptr
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::Dispatch<F, R, Ts...>(func, ptr(), Tag() - 1);
    }

    template <typename F>
    DFPBRT_CPU_GPU decltype(auto) Dispatch(F &&func) const {
        DCHECK(ptr());
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::Dispatch<F, R, Ts...>(func, ptr(), Tag() - 1);
    }

    template <typename F>
    decltype(auto) DispatchCPU(F &&func) {
        DCHECK(ptr());
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::DispatchCPU<F, R, Ts...>(func, ptr(), Tag() - 1);
    }

    template <typename F>
    decltype(auto) DispatchCPU(F &&func) const {
        DCHECK(ptr());
        using R = typename detail::ReturnTypeConst<F, Ts...>::type;
        return detail::DispatchCPU<F, R, Ts...>(func, ptr(), Tag() - 1);
    }

  private:
    static_assert(sizeof(uintptr_t) <= sizeof(uint64_t),
                  "Expected pointer size to be <= 64 bits");
    // TaggedPointer Private Members
    // x x x x x x x x ... x x
    //| 7 bits of tag ||57 bits of ptr|
    static constexpr int tagShift = 57;
    static constexpr int tagBits = 64 - tagShift;
    // ull: unsigned long long
    static constexpr uint64_t tagMask = ((1ull << tagBits) - 1) << tagShift;
    static constexpr uint64_t ptrMask = ~tagMask;
    uint64_t bits = 0;
};




}

#endif