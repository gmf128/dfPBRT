#ifndef DFPBRT_UTIL_COLOR_H
#define DFPBRT_UTIL_COLOR_H

#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/check.h>
#include <dfpbrt/util/math.h>
#include <dfpbrt/util/taggedptr.h>
#include <dfpbrt/util/vecmath.h>

// A special present from windgi.h on Windows...
#ifdef RGB
#undef RGB
#endif  // RGB

namespace dfpbrt{

// RGB Definition
class RGB {
  public:
    // RGB Public Methods
    RGB() = default;
    DFPBRT_CPU_GPU
    RGB(Float r, Float g, Float b) : r(r), g(g), b(b) {}

    DFPBRT_CPU_GPU
    RGB &operator+=(RGB s) {
        r += s.r;
        g += s.g;
        b += s.b;
        return *this;
    }
    DFPBRT_CPU_GPU
    RGB operator+(RGB s) const {
        RGB ret = *this;
        return ret += s;
    }

    DFPBRT_CPU_GPU
    RGB &operator-=(RGB s) {
        r -= s.r;
        g -= s.g;
        b -= s.b;
        return *this;
    }
    DFPBRT_CPU_GPU
    RGB operator-(RGB s) const {
        RGB ret = *this;
        return ret -= s;
    }
    DFPBRT_CPU_GPU
    friend RGB operator-(Float a, RGB s) { return {a - s.r, a - s.g, a - s.b}; }

    DFPBRT_CPU_GPU
    RGB &operator*=(RGB s) {
        r *= s.r;
        g *= s.g;
        b *= s.b;
        return *this;
    }
    DFPBRT_CPU_GPU
    RGB operator*(RGB s) const {
        RGB ret = *this;
        return ret *= s;
    }
    DFPBRT_CPU_GPU
    RGB operator*(Float a) const {
        DCHECK(!IsNaN(a));
        return {a * r, a * g, a * b};
    }
    DFPBRT_CPU_GPU
    RGB &operator*=(Float a) {
        DCHECK(!IsNaN(a));
        r *= a;
        g *= a;
        b *= a;
        return *this;
    }
    DFPBRT_CPU_GPU
    friend RGB operator*(Float a, RGB s) { return s * a; }

    DFPBRT_CPU_GPU
    RGB &operator/=(RGB s) {
        r /= s.r;
        g /= s.g;
        b /= s.b;
        return *this;
    }
    DFPBRT_CPU_GPU
    RGB operator/(RGB s) const {
        RGB ret = *this;
        return ret /= s;
    }
    DFPBRT_CPU_GPU
    RGB &operator/=(Float a) {
        DCHECK(!IsNaN(a));
        DCHECK_NE(a, 0);
        r /= a;
        g /= a;
        b /= a;
        return *this;
    }
    DFPBRT_CPU_GPU
    RGB operator/(Float a) const {
        RGB ret = *this;
        return ret /= a;
    }

    DFPBRT_CPU_GPU
    RGB operator-() const { return {-r, -g, -b}; }

    DFPBRT_CPU_GPU
    Float Average() const { return (r + g + b) / 3; }

    DFPBRT_CPU_GPU
    bool operator==(RGB s) const { return r == s.r && g == s.g && b == s.b; }
    DFPBRT_CPU_GPU
    bool operator!=(RGB s) const { return r != s.r || g != s.g || b != s.b; }
    DFPBRT_CPU_GPU
    Float operator[](int c) const {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return r;
        else if (c == 1)
            return g;
        return b;
    }
    DFPBRT_CPU_GPU
    Float &operator[](int c) {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return r;
        else if (c == 1)
            return g;
        return b;
    }

    std::string ToString() const;

    // RGB Public Members
    Float r = 0, g = 0, b = 0;
};

#define NOMINMAX
DFPBRT_CPU_GPU
inline RGB max(RGB a, RGB b) {
    return RGB(std::max(a.r, b.r), std::max(a.g, b.g), std::max(a.b, b.b));
}

DFPBRT_CPU_GPU
inline RGB Lerp(Float t, RGB s1, RGB s2) {
    return (1 - t) * s1 + t * s2;
}

// RGB Inline Functions
template <typename U, typename V>
DFPBRT_CPU_GPU inline RGB Clamp(RGB rgb, U min, V max) {
    return RGB(dfpbrt::Clamp(rgb.r, min, max), dfpbrt::Clamp(rgb.g, min, max),
               dfpbrt::Clamp(rgb.b, min, max));
}
DFPBRT_CPU_GPU inline RGB ClampZero(RGB rgb) {
    return RGB(std::max<Float>(0, rgb.r), std::max<Float>(0, rgb.g),
               std::max<Float>(0, rgb.b));
}

// XYZ Definition
class XYZ {
  public:
    // XYZ Public Methods
    XYZ() = default;
    DFPBRT_CPU_GPU
    XYZ(Float X, Float Y, Float Z) : X(X), Y(Y), Z(Z) {}

    DFPBRT_CPU_GPU
    Float Average() const { return (X + Y + Z) / 3; }

    DFPBRT_CPU_GPU
    Point2f xy() const { return Point2f(X / (X + Y + Z), Y / (X + Y + Z)); }

    DFPBRT_CPU_GPU
    static XYZ FromxyY(Point2f xy, Float Y = 1) {
        if (xy.y == 0)
            return XYZ(0, 0, 0);
        return XYZ(xy.x * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y);
    }

    DFPBRT_CPU_GPU
    XYZ &operator+=(const XYZ &s) {
        X += s.X;
        Y += s.Y;
        Z += s.Z;
        return *this;
    }
    DFPBRT_CPU_GPU
    XYZ operator+(const XYZ &s) const {
        XYZ ret = *this;
        return ret += s;
    }

    DFPBRT_CPU_GPU
    XYZ &operator-=(const XYZ &s) {
        X -= s.X;
        Y -= s.Y;
        Z -= s.Z;
        return *this;
    }
    DFPBRT_CPU_GPU
    XYZ operator-(const XYZ &s) const {
        XYZ ret = *this;
        return ret -= s;
    }
    DFPBRT_CPU_GPU
    friend XYZ operator-(Float a, const XYZ &s) { return {a - s.X, a - s.Y, a - s.Z}; }

    DFPBRT_CPU_GPU
    XYZ &operator*=(const XYZ &s) {
        X *= s.X;
        Y *= s.Y;
        Z *= s.Z;
        return *this;
    }
    DFPBRT_CPU_GPU
    XYZ operator*(const XYZ &s) const {
        XYZ ret = *this;
        return ret *= s;
    }
    DFPBRT_CPU_GPU
    XYZ operator*(Float a) const {
        DCHECK(!IsNaN(a));
        return {a * X, a * Y, a * Z};
    }
    DFPBRT_CPU_GPU
    XYZ &operator*=(Float a) {
        DCHECK(!IsNaN(a));
        X *= a;
        Y *= a;
        Z *= a;
        return *this;
    }

    DFPBRT_CPU_GPU
    XYZ &operator/=(const XYZ &s) {
        X /= s.X;
        Y /= s.Y;
        Z /= s.Z;
        return *this;
    }
    DFPBRT_CPU_GPU
    XYZ operator/(const XYZ &s) const {
        XYZ ret = *this;
        return ret /= s;
    }
    DFPBRT_CPU_GPU
    XYZ &operator/=(Float a) {
        DCHECK(!IsNaN(a));
        DCHECK_NE(a, 0);
        X /= a;
        Y /= a;
        Z /= a;
        return *this;
    }
    DFPBRT_CPU_GPU
    XYZ operator/(Float a) const {
        XYZ ret = *this;
        return ret /= a;
    }

    DFPBRT_CPU_GPU
    XYZ operator-() const { return {-X, -Y, -Z}; }

    DFPBRT_CPU_GPU
    bool operator==(const XYZ &s) const { return X == s.X && Y == s.Y && Z == s.Z; }
    DFPBRT_CPU_GPU
    bool operator!=(const XYZ &s) const { return X != s.X || Y != s.Y || Z != s.Z; }
    DFPBRT_CPU_GPU
    Float operator[](int c) const {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return X;
        else if (c == 1)
            return Y;
        return Z;
    }
    DFPBRT_CPU_GPU
    Float &operator[](int c) {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return X;
        else if (c == 1)
            return Y;
        return Z;
    }

    std::string ToString() const;

    // XYZ Public Members
    Float X = 0, Y = 0, Z = 0;
};

DFPBRT_CPU_GPU
inline XYZ operator*(Float a, const XYZ &s) {
    return s * a;
}

template <typename U, typename V>
DFPBRT_CPU_GPU inline XYZ Clamp(const XYZ &xyz, U min, V max) {
    return XYZ(dfpbrt::Clamp(xyz.X, min, max), dfpbrt::Clamp(xyz.Y, min, max),
               dfpbrt::Clamp(xyz.Z, min, max));
}

DFPBRT_CPU_GPU
inline XYZ ClampZero(const XYZ &xyz) {
    return XYZ(std::max<Float>(0, xyz.X), std::max<Float>(0, xyz.Y),
               std::max<Float>(0, xyz.Z));
}

DFPBRT_CPU_GPU
inline XYZ Lerp(Float t, const XYZ &s1, const XYZ &s2) {
    return (1 - t) * s1 + t * s2;
}

// RGBSigmoidPolynomial Definition
class RGBSigmoidPolynomial {
  public:
    // RGBSigmoidPolynomial Public Methods
    RGBSigmoidPolynomial() = default;
    DFPBRT_CPU_GPU
    RGBSigmoidPolynomial(Float c0, Float c1, Float c2) : c0(c0), c1(c1), c2(c2) {}
    std::string ToString() const;

    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const {
        return s(EvaluatePolynomial(lambda, c2, c1, c0));
    }

    DFPBRT_CPU_GPU
    Float MaxValue() const {
        Float result = std::max((*this)(360), (*this)(830));
        Float lambda = -c1 / (2 * c0);
        if (lambda >= 360 && lambda <= 830)
            result = std::max(result, (*this)(lambda));
        return result;
    }

  private:
    // RGBSigmoidPolynomial Private Methods
    DFPBRT_CPU_GPU
    static Float s(Float x) {
        if (IsInf(x))
            return x > 0 ? 1 : 0;
        return .5f + x / (2 * std::sqrt(1 + Sqr(x)));
    };

    // RGBSigmoidPolynomial Private Members
    Float c0, c1, c2;
};




}

#endif