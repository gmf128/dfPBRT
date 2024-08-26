#ifndef DFPBRT_UTIL_RNG_H
#define DFPBRT_UTIL_RNG_H

#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/check.h>
#include <dfpbrt/util/float.h>
#include <dfpbrt/util/hash.h>
#include <dfpbrt/util/math.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace dfpbrt {

// Random Number Declarations
#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

static int32_t _numeric_max = (std::numeric_limits<int32_t>::max)();
static int32_t _numeric_min = (std::numeric_limits<int32_t>::min)();

// RNG Definition
class RNG {
  public:
    // RNG Public Methods
    RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}
    RNG(uint64_t seqIndex, uint64_t offset) { SetSequence(seqIndex, offset); }
    RNG(uint64_t seqIndex) { SetSequence(seqIndex); }

    void SetSequence(uint64_t sequenceIndex, uint64_t offset);
    void SetSequence(uint64_t sequenceIndex) {
        SetSequence(sequenceIndex, MixBits(sequenceIndex));
    }

    template <typename T>
    T Uniform();

    template <std::integral T>
    T Uniform(T b) {
        T threshold = (~b + 1u) % b;
        while (true) {
            T r = Uniform<T>();
            if (r >= threshold)
                return r % b;
        }
    }

    void Advance(int64_t idelta);
    int64_t operator-(const RNG &other) const;

    std::string ToString() const;

  private:
    // RNG Private Members
    uint64_t state, inc;
};


// RNG Inline Method Definitions
template <typename T>
inline T RNG::Uniform() {
    return T::unimplemented;
}

template <>
inline uint32_t RNG::Uniform<uint32_t>();

template <>
inline uint32_t RNG::Uniform<uint32_t>() {
    uint64_t oldstate = state;
    state = oldstate * PCG32_MULT + inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

template <>
inline uint64_t RNG::Uniform<uint64_t>() {
    uint64_t v0 = Uniform<uint32_t>(), v1 = Uniform<uint32_t>();
    return (v0 << 32) | v1;
}

template <>
inline int32_t RNG::Uniform<int32_t>() {
    // https://stackoverflow.com/a/13208789
    uint32_t v = Uniform<uint32_t>();

    if (v <= _numeric_max)
        return int32_t(v);
    DCHECK(v>= _numeric_min);
    return int32_t(v - _numeric_min) +
           _numeric_min;
}

template <>
inline int64_t RNG::Uniform<int64_t>() {
    // https://stackoverflow.com/a/13208789
    uint64_t v = Uniform<uint64_t>();
    if (v <= _numeric_max)
        // Safe to type convert directly.
        return int64_t(v);
    DCHECK(v>= _numeric_min);
    return int64_t(v - _numeric_min) +
           _numeric_min;
}

inline void RNG::SetSequence(uint64_t sequenceIndex, uint64_t seed) {
    state = 0u;
    inc = (sequenceIndex << 1u) | 1u;
    Uniform<uint32_t>();
    state += seed;
    Uniform<uint32_t>();
}

template <>
inline float RNG::Uniform<float>() {
    return std::min<float>(OneMinusEpsilon, Uniform<uint32_t>() * 0x1p-32f);
}

template <>
inline double RNG::Uniform<double>() {
    return std::min<double>(OneMinusEpsilon, Uniform<uint64_t>() * 0x1p-64);
}

inline void RNG::Advance(int64_t idelta) {
    uint64_t curMult = PCG32_MULT, curPlus = inc, accMult = 1u;
    uint64_t accPlus = 0u, delta = (uint64_t)idelta;
    while (delta > 0) {
        if (delta & 1) {
            accMult *= curMult;
            accPlus = accPlus * curMult + curPlus;
        }
        curPlus = (curMult + 1) * curPlus;
        curMult *= curMult;
        delta /= 2;
    }
    state = accMult * state + accPlus;
}


inline int64_t RNG::operator-(const RNG &other) const {
    CHECK(inc == other.inc);
    uint64_t curMult = PCG32_MULT, curPlus = inc, curState = other.state;
    uint64_t theBit = 1u, distance = 0u;
    while (state != curState) {
        if ((state & theBit) != (curState & theBit)) {
            curState = curState * curMult + curPlus;
            distance |= theBit;
        }
        CHECK((state & theBit) == (curState & theBit));
        theBit <<= 1;
        curPlus = (curMult + 1ULL) * curPlus;
        curMult *= curMult;
    }
    return (int64_t)distance;
}

}  // namespace dfpbrt

#endif  // DFPBRT_UTIL_RNG_H
