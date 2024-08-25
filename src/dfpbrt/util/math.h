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

}

#endif