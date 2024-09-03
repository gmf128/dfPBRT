#ifndef DFPBRT_UTIL_MATH_H
#define DFPBRT_UTIL_MATH_H

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/float.h>
#include <dfpbrt/util/check.h>
#include <cmath>
#include <string>
#include <span>
#include <cassert>
#include <optional>

namespace dfpbrt{

// Mathematical Constants
constexpr Float ShadowEpsilon = 0.0001f;

constexpr Float Pi = 3.14159265358979323846;
constexpr Float InvPi = 0.31830988618379067154;
constexpr Float Inv2Pi = 0.15915494309189533577;
constexpr Float Inv4Pi = 0.07957747154594766788;
constexpr Float PiOver2 = 1.57079632679489661923;
constexpr Float PiOver4 = 0.78539816339744830961;
constexpr Float Sqrt2 = 1.41421356237309504880;



// Interval Definition
class Interval {
  public:
    // Interval Public Methods
    Interval() = default;

    
    explicit Interval(Float v) : low(v), high(v) {}
    constexpr Interval(Float low, Float high)
        : low((std::min)(low, high)), high((std::max)(low, high)) {}

    //Why static?
    static Interval FromValueAndError(Float v, Float err) {
        Interval i;
        if (err == 0)
            i.low = i.high = v;
        else {
            i.low = dfpbrt::SubRoundDown(v, err);
            i.high = dfpbrt::AddRoundUp(v, err);
        }
        return i;
    }

    Interval &operator=(Float v) {
        low = high = v;
        return *this;
    }

    
    Float UpperBound() const { return high; }
    
    Float LowerBound() const { return low; }
    
    Float Midpoint() const { return (low + high) / 2; }
   
    Float Width() const { return high - low; }

    
    Float operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? low : high;
    }
    
    //type change from Interval to Point is change to its midpoint
    explicit operator Float() const { return Midpoint(); }

    bool Exactly(Float v) const { return low == v && high == v; }

    bool operator==(Float v) const { return Exactly(v); }

    Interval operator-() const { return {-high, -low}; }

    Interval operator+(Interval i) const {
        return {AddRoundDown(low, i.low), AddRoundUp(high, i.high)};
    }

    Interval operator-(Interval i) const {
        return {SubRoundDown(low, i.high), SubRoundUp(high, i.low)};
    }

    Interval operator*(Interval i) const {
        Float lp[4] = {MulRoundDown(low, i.low), MulRoundDown(high, i.low),
                       MulRoundDown(low, i.high), MulRoundDown(high, i.high)};
        Float hp[4] = {MulRoundUp(low, i.low), MulRoundUp(high, i.low),
                       MulRoundUp(low, i.high), MulRoundUp(high, i.high)};
        return {(std::min)({lp[0], lp[1], lp[2], lp[3]}),
                (std::max)({hp[0], hp[1], hp[2], hp[3]})};
    }

    Interval operator/(Interval i) const;

    bool operator==(Interval i) const {
        return low == i.low && high == i.high;
    }

    bool operator!=(Float f) const { return f < low || f > high; }

    std::string ToString() const;

    Interval &operator+=(Interval i) {
        *this = Interval(*this + i);
        return *this;
    }

    Interval &operator-=(Interval i) {
        *this = Interval(*this - i);
        return *this;
    }
    Interval &operator*=(Interval i) {
        *this = Interval(*this * i);
        return *this;
    }
   Interval &operator/=(Interval i) {
        *this = Interval(*this / i);
        return *this;
    }
    
    Interval &operator+=(Float f) { return *this += Interval(f); }
    
    Interval &operator-=(Float f) { return *this -= Interval(f); }
    
    Interval &operator*=(Float f) {
        if (f > 0)
            *this = Interval(MulRoundDown(f, low), MulRoundUp(f, high));
        else
            *this = Interval(MulRoundDown(f, high), MulRoundUp(f, low));
        return *this;
    }
   
    Interval &operator/=(Float f) {
        if (f > 0)
            *this = Interval(DivRoundDown(low, f), DivRoundUp(high, f));
        else
            *this = Interval(DivRoundDown(high, f), DivRoundUp(low, f));
        return *this;
    }


    static const Interval Pi;


  private:
    //friend struct SOA<Interval>;
    // Interval Private Members
    Float low, high;
};


// CompensatedFloat Definition
struct CompensatedFloat {
  public:
    // CompensatedFloat Public Methods
    CompensatedFloat(Float v, Float err = 0) : v(v), err(err) {}

    explicit operator float() const { return v + err; }
    
    explicit operator double() const { return double(v) + double(err); }
    std::string ToString() const;

    Float v, err;
};


inline Float Radians(Float deg) {
    return (Pi / 180) * deg;
}
inline Float Degrees(Float rad) {
    return (180 / Pi) * rad;
}

template <typename T>
inline constexpr T Sqr(T v) {
    return v * v;
}

template <typename T, typename U, typename V>
inline constexpr T Clamp(T val, U low, V high) {
    if (val < low)
        return T(low);
    else if (val > high)
        return T(high);
    else
        return val;
}

// Would be nice to allow Float to be a template type here, but it is tricky:
// https://stackoverflow.com/questions/5101516/why-function-template-cannot-be-partially-specialized
template <int n>
inline constexpr float Pow(float v) {
    if constexpr (n < 0)
        return 1 / Pow<-n>(v);
    float n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

template <>
inline constexpr float Pow<1>(float v) {
    return v;
}
template <>
inline constexpr float Pow<0>(float v) {
    return 1;
}

template <int n>
inline constexpr double Pow(double v) {
    if constexpr (n < 0)
        return 1 / Pow<-n>(v);
    double n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

template <>
inline constexpr double Pow<1>(double v) {
    return v;
}

template <>
inline constexpr double Pow<0>(double v) {
    return 1;
}

template <typename Float, typename C>
inline constexpr Float EvaluatePolynomial(Float t, C c) {
    return c;
}

template <typename Float, typename C, typename... Args>
inline constexpr Float EvaluatePolynomial(Float t, C c, Args... cRemaining) {
    return FMA(t, EvaluatePolynomial(t, cRemaining...), c);
}

inline Float SinXOverX(Float x) {
    if (1 - x * x == 1)
        return 1;
    return std::sin(x) / x;
}

inline float SafeASin(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1, 1));
}
inline float SafeACos(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(Clamp(x, -1, 1));
}

inline double SafeASin(double x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1, 1));
}

inline double SafeACos(double x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(Clamp(x, -1, 1));
}

inline float SafeSqrt(float x){
    DCHECK(x >= -1e-3); // not too negative
    return std::sqrt((std::max<float>)({0, x}));
}\

inline double SafeSqrt(double x){
    DCHECK(x >= -1e-3);
    return std::sqrt((std::max<double>)({0, x}));
}

inline Float Log2(Float x) {
    const Float invLog2 = 1.442695040888963387004650940071;
    return std::log(x) * invLog2;
}

inline int Log2Int(float v) {
    DCHECK(v > 0);
    if (v < 1)
        return -Log2Int(1 / v);
    // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    // (With an additional check of the significant to get round-to-nearest
    // rather than round down.)
    // midsignif = Significand(std::pow(2., 1.5))
    // i.e. grab the significand of a value halfway between two exponents,
    // in log space.
    const uint32_t midsignif = 0b00000000001101010000010011110011;
    return Exponent(v) + ((Significand(v) >= midsignif) ? 1 : 0);
}

inline int Log2Int(double v) {
    DCHECK(v > 0);
    if (v < 1)
        return -Log2Int(1 / v);
    // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    // (With an additional check of the significant to get round-to-nearest
    // rather than round down.)
    // midsignif = Significand(std::pow(2., 1.5))
    // i.e. grab the significand of a value halfway between two exponents,
    // in log space.
    const uint64_t midsignif = 0b110101000001001111001100110011111110011101111001101;
    return Exponent(v) + ((Significand(v) >= midsignif) ? 1 : 0);
}

inline int Log2Int(uint32_t v) {
    unsigned long lz = 0;
    if (_BitScanReverse(&lz, v))
        return lz;
    return 0;
}

inline int Log2Int(int32_t v) {
    return Log2Int((uint32_t)v);
}

inline int Log2Int(uint64_t v) {
    unsigned long lz = 0;
    _BitScanReverse64(&lz, v);
// _WIN64
    return lz;
}

inline int Log2Int(int64_t v) {
    return Log2Int((uint64_t)v);
}

template <typename T>
inline int Log4Int(T v) {
    return Log2Int(v) / 2;
}

// https://stackoverflow.com/a/10792321
inline float FastExp(float x) {
    // Compute $x'$ such that $\roman{e}^x = 2^{x'}$
    float xp = x * 1.442695041f;

    // Find integer and fractional components of $x'$
    float fxp = std::floor(xp), f = xp - fxp;
    int i = (int)fxp;

    // Evaluate polynomial approximation of $2^f$
    float twoToF = EvaluatePolynomial(f, 1.f, 0.695556856f, 0.226173572f, 0.0781455737f);

    // Scale $2^f$ by $2^i$ and return final result
    int exponent = Exponent(twoToF) + i;
    if (exponent < -126)
        return 0;
    if (exponent > 127)
        return Infinity_;
    uint32_t bits = FloatToBits(twoToF);
    bits &= 0b10000000011111111111111111111111u;
    bits |= (exponent + 127) << 23;
    return BitsToFloat(bits);
}


template <typename Ta, typename Tb, typename Tc, typename Td>
inline auto DifferenceOfProducts(Ta a, Tb b, Tc c, Td d) {
    //calculate ab-cd with high precision
    auto cd = c * d;
    auto differenceOfProducts = std::fma(a, b, -cd);
    auto error = std::fma(-c, d, cd);
    return differenceOfProducts + error;
}

template <typename Ta, typename Tb, typename Tc, typename Td>
inline auto SumOfProducts(Ta a, Tb b, Tc c, Td d) {
    //calculte ab+cd with high precision
    auto cd = c * d;
    auto sumOfProducts = FMA(a, b, cd);
    auto error = FMA(c, d, -cd);
    return sumOfProducts + error;
}

inline CompensatedFloat TwoProd(Float a, Float b) {
    Float ab = a * b;
    // error of a*b: FMA(a, b, -a*b)
    return {ab, FMA(a, b, -ab)};
}

inline CompensatedFloat TwoSum(Float a, Float b) {
    Float s = a + b, delta = s - a;
    // error: a - s + delta + b - delta = a + b - s
    // but the meaning of division of （a+b-s) into (a - (s - delta) + (b - delta)) where delta = s - a remains to be explored
    //TODO:
    return {s, (a - (s - delta)) + (b - delta)};
}

namespace internal {
// InnerProduct Helper Functions
template <typename Float>
inline CompensatedFloat InnerProduct(Float a, Float b) {
    return TwoProd(a, b);
}

// Accurate dot products with FMA: Graillat et al.,
// https://www-pequan.lip6.fr/~graillat/papers/posterRNC7.pdf
//
// Accurate summation, dot product and polynomial evaluation in complex
// floating point arithmetic, Graillat and Menissier-Morain.
template <typename Float, typename... T>
inline CompensatedFloat InnerProduct(Float a, Float b, T... terms) {
    CompensatedFloat ab = TwoProd(a, b);
    CompensatedFloat tp = InnerProduct(terms...);
    CompensatedFloat sum = TwoSum(ab.v, tp.v);
    return {sum.v, ab.err + (tp.err + sum.err)};
}

}  // namespace internal

// template <typename... T>
// inline std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>...>, Float>
// InnerProduct(T... terms) {
//     CompensatedFloat ip = internal::InnerProduct(terms...);
//     return Float(ip);
// }
//another version

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <Arithmetic... T>
inline Float InnerProduct(T... Terms){
    CompensatedFloat ip = internal::InnerProduct(Terms...);
    return Float(ip);
}

inline bool Quadratic(float a, float b, float c, float *t0, float *t1) {
    // solve the quadratic equation ax^2 + bx + c = 0 and two roots are t0 and t1
    // Handle case of $a=0$ for quadratic solution
    if (a == 0) {
        if (b == 0)
            return false;
        *t0 = *t1 = -c / b;
        return true;
    }

    // Find quadratic discriminant
    float discrim = DifferenceOfProducts(b, b, 4 * a, c);
    if (discrim < 0)
        return false;
    float rootDiscrim = std::sqrt(discrim);

    // Compute quadratic _t_ values
    float q = -0.5f * (b + std::copysign(rootDiscrim, b));
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1)
        std::swap(*t0, *t1);

    return true;
}


inline bool Quadratic(double a, double b, double c, double *t0, double *t1) {
    // Find quadratic discriminant
    double discrim = DifferenceOfProducts(b, b, 4 * a, c);
    if (discrim < 0)
        return false;
    double rootDiscrim = std::sqrt(discrim);

    if (a == 0) {
        *t0 = *t1 = -c / b;
        return true;
    }

    // Compute quadratic _t_ values
    double q = -0.5 * (b + std::copysign(rootDiscrim, b));
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1)
        std::swap(*t0, *t1);
    return true;
}

// Math Inline Functions
DFPBRT_CPU_GPU inline Float Lerp(Float x, Float a, Float b) {
    return (1 - x) * a + x * b;
}

template <typename Predicate>
DFPBRT_CPU_GPU inline size_t FindInterval(size_t sz, const Predicate &pred) {
    using ssize_t = std::make_signed_t<size_t>;
    ssize_t size = (ssize_t)sz - 2, first = 1;
    while (size > 0) {
        // Evaluate predicate at midpoint and update _first_ and _size_
        size_t half = (size_t)size >> 1, middle = first + half;
        bool predResult = pred(middle);
        first = predResult ? middle + 1 : first;
        size = predResult ? size - (half + 1) : half;
    }
    return (size_t)Clamp((ssize_t)first - 1, 0, sz - 2);
}

// Interval Inline Functions
inline bool InRange(Float v, Interval i) {
    return v >= i.LowerBound() && v <= i.UpperBound();
}
inline bool InRange(Interval a, Interval b) {
    return a.LowerBound() <= b.UpperBound() && a.UpperBound() >= b.LowerBound();
}

inline Interval Interval::operator/(Interval i) const {
    if (InRange(0, i))
        // The interval we're dividing by straddles zero, so just
        // return an interval of everything.
        return Interval(-Infinity_, Infinity_);

    Float lowQuot[4] = {DivRoundDown(low, i.low), DivRoundDown(high, i.low),
                        DivRoundDown(low, i.high), DivRoundDown(high, i.high)};
    Float highQuot[4] = {DivRoundUp(low, i.low), DivRoundUp(high, i.low),
                         DivRoundUp(low, i.high), DivRoundUp(high, i.high)};
    return {(std::min)({lowQuot[0], lowQuot[1], lowQuot[2], lowQuot[3]}),
            (std::max)({highQuot[0], highQuot[1], highQuot[2], highQuot[3]})};
}

inline Interval Sqr(Interval i) {
    Float alow = std::abs(i.LowerBound()), ahigh = std::abs(i.UpperBound());
    if (alow > ahigh)
        std::swap(alow, ahigh);
    if (InRange(0, i))
        return Interval(0, MulRoundUp(ahigh, ahigh));
    return Interval(MulRoundDown(alow, alow), MulRoundUp(ahigh, ahigh));
}

inline Interval MulPow2(Float s, Interval i);
inline Interval MulPow2(Interval i, Float s);

inline Interval operator+(Float f, Interval i) {
    return Interval(f) + i;
}

inline Interval operator-(Float f, Interval i) {
    return Interval(f) - i;
}

inline Interval operator*(Float f, Interval i) {
    if (f > 0)
        return Interval(MulRoundDown(f, i.LowerBound()), MulRoundUp(f, i.UpperBound()));
    else
        return Interval(MulRoundDown(f, i.UpperBound()), MulRoundUp(f, i.LowerBound()));
}
inline Interval operator/(Float f, Interval i) {
    if (InRange(0, i))
        // The interval we're dividing by straddles zero, so just
        // return an interval of everything.
        return Interval(-Infinity_, Infinity_);

    if (f > 0)
        return Interval(DivRoundDown(f, i.UpperBound()), DivRoundUp(f, i.LowerBound()));
    else
        return Interval(DivRoundDown(f, i.LowerBound()), DivRoundUp(f, i.UpperBound()));
}

inline Interval operator+(Interval i, Float f) {
    return i + Interval(f);
}

inline Interval operator-(Interval i, Float f) {
    return i - Interval(f);
}

inline Interval operator*(Interval i, Float f) {
    if (f > 0)
        return Interval(MulRoundDown(f, i.LowerBound()), MulRoundUp(f, i.UpperBound()));
    else
        return Interval(MulRoundDown(f, i.UpperBound()), MulRoundUp(f, i.LowerBound()));
}

inline Interval operator/(Interval i, Float f) {
    if (f == 0)
        return Interval(-Infinity_, Infinity_);

    if (f > 0)
        return Interval(DivRoundDown(i.LowerBound(), f), DivRoundUp(i.UpperBound(), f));
    else
        return Interval(DivRoundDown(i.UpperBound(), f), DivRoundUp(i.LowerBound(), f));
}

inline Float Floor(Interval i) {
    return std::floor(i.LowerBound());
}

inline Float Ceil(Interval i) {
    return std::ceil(i.UpperBound());
}

inline Float floor(Interval i) {
    return Floor(i);
}

inline Float ceil(Interval i) {
    return Ceil(i);
}

inline Float Min(Interval a, Interval b) {
    return (std::min)(a.LowerBound(), b.LowerBound());
}

inline Float Max(Interval a, Interval b) {
    return (std::max)(a.UpperBound(), b.UpperBound());
}


inline Interval Sqrt(Interval i) {
    return {SqrtRoundDown(i.LowerBound()), SqrtRoundUp(i.UpperBound())};
}

inline Interval sqrt(Interval i) {
    return Sqrt(i);
}

inline Interval FMA(Interval a, Interval b, Interval c) {
    Float low = (std::min)({FMARoundDown(a.LowerBound(), b.LowerBound(), c.LowerBound()),
                          FMARoundDown(a.UpperBound(), b.LowerBound(), c.LowerBound()),
                          FMARoundDown(a.LowerBound(), b.UpperBound(), c.LowerBound()),
                          FMARoundDown(a.UpperBound(), b.UpperBound(), c.LowerBound())});
    Float high = (std::max)({FMARoundUp(a.LowerBound(), b.LowerBound(), c.UpperBound()),
                           FMARoundUp(a.UpperBound(), b.LowerBound(), c.UpperBound()),
                           FMARoundUp(a.LowerBound(), b.UpperBound(), c.UpperBound()),
                           FMARoundUp(a.UpperBound(), b.UpperBound(), c.UpperBound())});
    return Interval(low, high);
}

inline Interval DifferenceOfProducts(Interval a, Interval b, Interval c,
                                                  Interval d) {
    Float ab[4] = {a.LowerBound() * b.LowerBound(), a.UpperBound() * b.LowerBound(),
                   a.LowerBound() * b.UpperBound(), a.UpperBound() * b.UpperBound()};
    Float abLow = (std::min)({ab[0], ab[1], ab[2], ab[3]});
    Float abHigh = (std::max)({ab[0], ab[1], ab[2], ab[3]});
    int abLowIndex = abLow == ab[0] ? 0 : (abLow == ab[1] ? 1 : (abLow == ab[2] ? 2 : 3));
    int abHighIndex =
        abHigh == ab[0] ? 0 : (abHigh == ab[1] ? 1 : (abHigh == ab[2] ? 2 : 3));

    Float cd[4] = {c.LowerBound() * d.LowerBound(), c.UpperBound() * d.LowerBound(),
                   c.LowerBound() * d.UpperBound(), c.UpperBound() * d.UpperBound()};
    Float cdLow = (std::min)({cd[0], cd[1], cd[2], cd[3]});
    Float cdHigh = (std::max)({cd[0], cd[1], cd[2], cd[3]});
    int cdLowIndex = cdLow == cd[0] ? 0 : (cdLow == cd[1] ? 1 : (cdLow == cd[2] ? 2 : 3));
    int cdHighIndex =
        cdHigh == cd[0] ? 0 : (cdHigh == cd[1] ? 1 : (cdHigh == cd[2] ? 2 : 3));

    // Invert cd Indices since it's subtracted...
    Float low = DifferenceOfProducts(a[abLowIndex & 1], b[abLowIndex >> 1],
                                     c[cdHighIndex & 1], d[cdHighIndex >> 1]);
    Float high = DifferenceOfProducts(a[abHighIndex & 1], b[abHighIndex >> 1],
                                      c[cdLowIndex & 1], d[cdLowIndex >> 1]);
    DCHECK(low <= high);

    return {NextFloatDown(NextFloatDown(low)), NextFloatUp(NextFloatUp(high))};
}

inline Interval SumOfProducts(Interval a, Interval b, Interval c,
                                           Interval d) {
    return DifferenceOfProducts(a, b, -c, d);
}

inline Interval MulPow2(Float s, Interval i) {
    return MulPow2(i, s);
}

inline Interval MulPow2(Interval i, Float s) {
    Float as = std::abs(s);
    if (as < 1)
        DCHECK(1 / as == 1ull << Log2Int(1 / as));
    else
        DCHECK(as == 1ull << Log2Int(as));

    // Multiplication by powers of 2 is exaact
    return Interval((std::min)(i.LowerBound() * s, i.UpperBound() * s),
                    (std::max)(i.LowerBound() * s, i.UpperBound() * s));
}

inline Interval Abs(Interval i) {
    if (i.LowerBound() >= 0)
        // The entire interval is greater than zero, so we're all set.
        return i;
    else if (i.UpperBound() <= 0)
        // The entire interval is less than zero.
        return Interval(-i.UpperBound(), -i.LowerBound());
    else
        // The interval straddles zero.
        return Interval(0, (std::max)(-i.LowerBound(), i.UpperBound()));
}
inline Interval abs(Interval i) {
    return Abs(i);
}

inline Interval ACos(Interval i) {
    Float low = std::acos(std::min<Float>(1, i.UpperBound()));
    Float high = std::acos(std::max<Float>(-1, i.LowerBound()));

    return Interval(std::max<Float>(0, NextFloatDown(low)), NextFloatUp(high));
}

inline Interval Sin(Interval i) {
    CHECK(i.LowerBound()>= -1e-16);
    CHECK(i.UpperBound()<= 2.0001 * Pi);
    Float low = std::sin(std::max<Float>(0, i.LowerBound()));
    Float high = std::sin(i.UpperBound());
    if (low > high)
        std::swap(low, high);
    low = std::max<Float>(-1, NextFloatDown(low));
    high = std::min<Float>(1, NextFloatUp(high));
    if (InRange(Pi / 2, i))
        high = 1;
    if (InRange((3.f / 2.f) * Pi, i))
        low = -1;

    return Interval(low, high);
}

inline Interval Cos(Interval i) {
    CHECK(i.LowerBound()>= -1e-16);
    CHECK(i.UpperBound() <= 2.0001 * Pi);
    Float low = std::cos(std::max<Float>(0, i.LowerBound()));
    Float high = std::cos(i.UpperBound());
    if (low > high)
        std::swap(low, high);
    low = std::max<Float>(-1, NextFloatDown(low));
    high = std::min<Float>(1, NextFloatUp(high));
    if (InRange(Pi, i))
        low = -1;

    return Interval(low, high);
}

inline bool Quadratic(Interval a, Interval b, Interval c, Interval *t0,
                                   Interval *t1) {
    // Find quadratic discriminant
    Interval discrim = DifferenceOfProducts(b, b, MulPow2(4, a), c);
    if (discrim.LowerBound() < 0)
        return false;
    Interval floatRootDiscrim = Sqrt(discrim);

    // Compute quadratic _t_ values
    Interval q;
    if ((Float)b < 0)
        q = MulPow2(-.5, b - floatRootDiscrim);
    else
        q = MulPow2(-.5, b + floatRootDiscrim);
    *t0 = q / a;
    *t1 = c / q;
    if (t0->LowerBound() > t1->LowerBound())
        std::swap(*t0, *t1);
    return true;
}

inline Interval SumSquares(Interval i) {
    return Sqr(i);
}

template <typename... Args>
inline Interval SumSquares(Interval i, Args... args) {
    Interval ss = FMA(i, i, SumSquares(args...));
    return Interval(std::max<Float>(0, ss.LowerBound()), ss.UpperBound());
}



namespace {

template <int N>
inline void init(Float m[N][N], int i, int j) {}

template <int N, typename... Args>
inline void init(Float m[N][N], int i, int j, Float v, Args... args) {
    //meta programming
    m[i][j] = v;
    if (++j == N) {
        ++i;
        j = 0;
    }
    //each iteration consumes a float number
    init<N>(m, i, j, args...);
}

template <int N>
inline void initDiag(Float m[N][N], int i) {}

template <int N, typename... Args>
inline void initDiag(Float m[N][N], int i, Float v, Args... args) {
    m[i][i] = v;
    initDiag<N>(m, i + 1, args...);
}

}  // namespace

// Definition and implemention of Squared matrix
// SquareMatrix Definition
template <int N>
class SquareMatrix {
  public:
    // SquareMatrix Public Methods
    static SquareMatrix Zero() {
        SquareMatrix m;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m.m[i][j] = 0;
        return m;
    }

    SquareMatrix() {
        //default initialization: identity matrix
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] = (i == j) ? 1 : 0;
    }
    
    SquareMatrix(const Float mat[N][N]) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] = mat[i][j];
    }
    
    SquareMatrix(std::span<const Float> t);
    template <typename... Args>
    SquareMatrix(Float v, Args... args) {
        //at least one parameter
        //parameter pack: 0 or more
        //float v, Args... args : 1 or more
        static_assert(1 + sizeof...(Args) == N * N,
                      "Incorrect number of values provided to SquareMatrix constructor");
        init<N>(m, 0, 0, v, args...);
    }
    template <typename... Args>
    static SquareMatrix Diag(Float v, Args... args) {
        static_assert(1 + sizeof...(Args) == N,
                      "Incorrect number of values provided to SquareMatrix::Diag");
        SquareMatrix m;
        initDiag<N>(m.m, 0, v, args...);
        return m;
    }

    SquareMatrix operator+(const SquareMatrix &m) const {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] += m.m[i][j];
        return r;
    }

    SquareMatrix operator*(Float s) const {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] *= s;
        return r;
    }
    
    SquareMatrix operator/(Float s) const {
        DCHECK(s != 0);
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] /= s;
        return r;
    }

    
    bool operator==(const SquareMatrix<N> &m2) const {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (m[i][j] != m2.m[i][j])
                    return false;
        return true;
    }

   
    bool operator!=(const SquareMatrix<N> &m2) const {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (m[i][j] != m2.m[i][j])
                    return true;
        return false;
    }

   
    bool operator<(const SquareMatrix<N> &m2) const {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                if (m[i][j] < m2.m[i][j])
                    return true;
                if (m[i][j] > m2.m[i][j])
                    return false;
            }
        return false;
    }

    bool IsIdentity() const;

    std::string ToString() const;

    
    std::span<const Float> operator[](int i) const { return m[i]; }
    
    std::span<Float> operator[](int i) { return std::span<Float>(m[i]); }

  private:
    Float m[N][N];
};

// SquareMatrix Inline Methods
template <int N>
inline bool SquareMatrix<N>::IsIdentity() const {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                if (m[i][j] != 1)
                    return false;
            } else if (m[i][j] != 0)
                return false;
        }
    return true;
}

// SquareMatrix Inline Functions
template <int N>
inline SquareMatrix<N> operator*(Float s, const SquareMatrix<N> &m) {
    return m * s;
}

template <typename Tresult, int N, typename T>
inline Tresult Mul(const SquareMatrix<N> &m, const T &v) {
    Tresult result;
    for (int i = 0; i < N; ++i) {
        result[i] = 0;
        for (int j = 0; j < N; ++j)
            result[i] += m[i][j] * v[j];
    }
    return result;
}

template <int N>
Float Determinant(const SquareMatrix<N> &m);

template <>
inline Float Determinant(const SquareMatrix<3> &m) {
    Float minor12 = DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    Float minor02 = DifferenceOfProducts(m[1][0], m[2][2], m[1][2], m[2][0]);
    Float minor01 = DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    return FMA(m[0][2], minor01,
               DifferenceOfProducts(m[0][0], minor12, m[0][1], minor02));
}

template <int N>
inline SquareMatrix<N> Transpose(const SquareMatrix<N> &m);

// std::optional<T> is used to store a value that maybe not exist. (from Cpp17)
// Here is a good place to use std::optional since a singular matrix has no inverse matrix.
template <int N>
std::optional<SquareMatrix<N>> Inverse(const SquareMatrix<N> &);

template <int N>
SquareMatrix<N> InvertOrExit(const SquareMatrix<N> &m) {
    //If not exist, exit the program(since CHECK not passed)
    std::optional<SquareMatrix<N>> inv = Inverse(m);
    CHECK(inv.has_value());
    return *inv;
}

template <int N>
inline SquareMatrix<N> Transpose(const SquareMatrix<N> &m) {
    SquareMatrix<N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            r[i][j] = m[j][i];
    return r;
}

template <>
inline std::optional<SquareMatrix<3>> Inverse(const SquareMatrix<3> &m) {
    Float det = Determinant(m);
    if (det == 0)
        return {};
    Float invDet = 1 / det;

    SquareMatrix<3> r;

    r[0][0] = invDet * DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    r[1][0] = invDet * DifferenceOfProducts(m[1][2], m[2][0], m[1][0], m[2][2]);
    r[2][0] = invDet * DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    r[0][1] = invDet * DifferenceOfProducts(m[0][2], m[2][1], m[0][1], m[2][2]);
    r[1][1] = invDet * DifferenceOfProducts(m[0][0], m[2][2], m[0][2], m[2][0]);
    r[2][1] = invDet * DifferenceOfProducts(m[0][1], m[2][0], m[0][0], m[2][1]);
    r[0][2] = invDet * DifferenceOfProducts(m[0][1], m[1][2], m[0][2], m[1][1]);
    r[1][2] = invDet * DifferenceOfProducts(m[0][2], m[1][0], m[0][0], m[1][2]);
    r[2][2] = invDet * DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);

    return r;
}

template <int N, typename T>
inline T operator*(const SquareMatrix<N> &m, const T &v) {
    return Mul<T>(m, v);
}

template <>
inline SquareMatrix<4> operator*(const SquareMatrix<4> &m1,
                                              const SquareMatrix<4> &m2) {
    SquareMatrix<4> r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            //accurate value of multi-fma
            r[i][j] = InnerProduct(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2],
                                   m2[2][j], m1[i][3], m2[3][j]);
    return r;
}

template <>
inline SquareMatrix<3> operator*(const SquareMatrix<3> &m1,
                                              const SquareMatrix<3> &m2) {
    SquareMatrix<3> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i][j] =
                InnerProduct(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2], m2[2][j]);
    return r;
}

template <int N>
inline SquareMatrix<N> operator*(const SquareMatrix<N> &m1,
                                              const SquareMatrix<N> &m2) {
    SquareMatrix<N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            r[i][j] = 0;
            for (int k = 0; k < N; ++k)
                r[i][j] = FMA(m1[i][k], m2[k][j], r[i][j]);
        }
    return r;
}

template <int N>
inline SquareMatrix<N>::SquareMatrix(std::span<const Float> t) {
    CHECK(N * N == t.size());
    for (int i = 0; i < N * N; ++i)
        m[i / N][i % N] = t[i];
}

template <int N>
SquareMatrix<N> operator*(const SquareMatrix<N> &m1,
                                       const SquareMatrix<N> &m2);

template <>
inline Float Determinant(const SquareMatrix<1> &m) {
    return m[0][0];
}

template <>
inline Float Determinant(const SquareMatrix<2> &m) {
    return DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);
}

template <>
inline Float Determinant(const SquareMatrix<4> &m) {
    Float s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
    Float s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
    Float s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);

    Float s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
    Float s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
    Float s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);

    Float c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
    Float c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
    Float c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);

    Float c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
    Float c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
    Float c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);

    return (DifferenceOfProducts(s0, c5, s1, c4) + DifferenceOfProducts(s2, c3, -s3, c2) +
            DifferenceOfProducts(s5, c0, s4, c1));
}

template <int N>
inline Float Determinant(const SquareMatrix<N> &m) {
    SquareMatrix<N - 1> sub;
    Float det = 0;
    // Inefficient, but we don't currently use N>4 anyway..
    for (int i = 0; i < N; ++i) {
        // Sub-matrix without row 0 and column i
        for (int j = 0; j < N - 1; ++j)
            for (int k = 0; k < N - 1; ++k)
                sub[j][k] = m[j + 1][k < i ? k : k + 1];

        Float sign = (i & 1) ? -1 : 1;
        det += sign * m[0][i] * Determinant(sub);
    }
    return det;
}

template <>
inline std::optional<SquareMatrix<4>> Inverse(const SquareMatrix<4> &m) {
    // Via: https://github.com/google/ion/blob/master/ion/math/matrixutils.cc,
    // (c) Google, Apache license.

    // For 4x4 do not compute the adjugate as the transpose of the cofactor
    // matrix, because this results in extra work. Several calculations can be
    // shared across the sub-determinants.
    //
    // This approach is explained in David Eberly's Geometric Tools book,
    // excerpted here:
    //   http://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    Float s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
    Float s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
    Float s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);

    Float s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
    Float s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
    Float s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);

    Float c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
    Float c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
    Float c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);

    Float c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
    Float c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
    Float c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);

    Float determinant = InnerProduct(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
    if (determinant == 0)
        return {};
    Float s = 1 / determinant;

    Float inv[4][4] = {{s * InnerProduct(m[1][1], c5, m[1][3], c3, -m[1][2], c4),
                        s * InnerProduct(-m[0][1], c5, m[0][2], c4, -m[0][3], c3),
                        s * InnerProduct(m[3][1], s5, m[3][3], s3, -m[3][2], s4),
                        s * InnerProduct(-m[2][1], s5, m[2][2], s4, -m[2][3], s3)},

                       {s * InnerProduct(-m[1][0], c5, m[1][2], c2, -m[1][3], c1),
                        s * InnerProduct(m[0][0], c5, m[0][3], c1, -m[0][2], c2),
                        s * InnerProduct(-m[3][0], s5, m[3][2], s2, -m[3][3], s1),
                        s * InnerProduct(m[2][0], s5, m[2][3], s1, -m[2][2], s2)},

                       {s * InnerProduct(m[1][0], c4, m[1][3], c0, -m[1][1], c2),
                        s * InnerProduct(-m[0][0], c4, m[0][1], c2, -m[0][3], c0),
                        s * InnerProduct(m[3][0], s4, m[3][3], s0, -m[3][1], s2),
                        s * InnerProduct(-m[2][0], s4, m[2][1], s2, -m[2][3], s0)},

                       {s * InnerProduct(-m[1][0], c3, m[1][1], c1, -m[1][2], c0),
                        s * InnerProduct(m[0][0], c3, m[0][2], c0, -m[0][1], c1),
                        s * InnerProduct(-m[3][0], s3, m[3][1], s1, -m[3][2], s0),
                        s * InnerProduct(m[2][0], s3, m[2][2], s0, -m[2][1], s1)}};

    return SquareMatrix<4>(inv);
}

extern template class SquareMatrix<2>;
extern template class SquareMatrix<3>;
extern template class SquareMatrix<4>;

Point2f WrapEqualAreaSquare(Point2f uv);
Point2f EqualAreaSphereToSquare(Vector3f d);
Vector3f EqualAreaSquareToSphere(Point2f p);

}

#endif