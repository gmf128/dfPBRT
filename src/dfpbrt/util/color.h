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

// RGBToSpectrumTable Definition
class RGBToSpectrumTable {
  public:
    // RGBToSpectrumTable Public Constants
    static constexpr int res = 64;

    using CoefficientArray = float[3][res][res][res][3];

    // RGBToSpectrumTable Public Methods
    RGBToSpectrumTable(const float *zNodes, const CoefficientArray *coeffs)
        : zNodes(zNodes), coeffs(coeffs) {}

    DFPBRT_CPU_GPU
    RGBSigmoidPolynomial operator()(RGB rgb) const;

    static void Init(Allocator alloc);

    static const RGBToSpectrumTable *sRGB;
    static const RGBToSpectrumTable *DCI_P3;
    static const RGBToSpectrumTable *Rec2020;
    static const RGBToSpectrumTable *ACES2065_1;

    std::string ToString() const;

  private:
    // RGBToSpectrumTable Private Members
    const float *zNodes;
    const CoefficientArray *coeffs;
};


// ColorEncoding Definitions
class LinearColorEncoding;
class sRGBColorEncoding;
class GammaColorEncoding;

class ColorEncoding
    : public TaggedPointer<LinearColorEncoding, sRGBColorEncoding, GammaColorEncoding> {
  public:
    using TaggedPointer::TaggedPointer;
    // ColorEncoding Interface
    DFPBRT_CPU_GPU inline void ToLinear(std::span<const uint8_t> vin,
                                      std::span<Float> vout) const;
    DFPBRT_CPU_GPU inline void FromLinear(std::span<const Float> vin,
                                        std::span<uint8_t> vout) const;

    DFPBRT_CPU_GPU inline Float ToFloatLinear(Float v) const;

    std::string ToString() const;

    static const ColorEncoding Get(const std::string &name, Allocator alloc);

    static ColorEncoding Linear;
    static ColorEncoding sRGB;

    static void Init(Allocator alloc);
};

class LinearColorEncoding {
  public:
    DFPBRT_CPU_GPU
    void ToLinear(std::span<const uint8_t> vin, std::span<Float> vout) const {
        DCHECK_EQ(vin.size(), vout.size());
        for (size_t i = 0; i < vin.size(); ++i)
            vout[i] = vin[i] / 255.f;
    }

    DFPBRT_CPU_GPU
    Float ToFloatLinear(Float v) const { return v; }

    DFPBRT_CPU_GPU
    void FromLinear(std::span<const Float> vin, std::span<uint8_t> vout) const {
        DCHECK_EQ(vin.size(), vout.size());
        for (size_t i = 0; i < vin.size(); ++i)
            vout[i] = uint8_t(Clamp(vin[i] * 255.f + 0.5f, 0, 255));
    }

    std::string ToString() const { return "[ LinearColorEncoding ]"; }
};

class sRGBColorEncoding {
  public:
    // sRGBColorEncoding Public Methods
    DFPBRT_CPU_GPU
    void ToLinear(std::span<const uint8_t> vin, std::span<Float> vout) const;
    DFPBRT_CPU_GPU
    Float ToFloatLinear(Float v) const;
    DFPBRT_CPU_GPU
    void FromLinear(std::span<const Float> vin, std::span<uint8_t> vout) const;

    std::string ToString() const { return "[ sRGBColorEncoding ]"; }
};

class GammaColorEncoding {
  public:
    DFPBRT_CPU_GPU
    GammaColorEncoding(Float gamma);

    DFPBRT_CPU_GPU
    void ToLinear(std::span<const uint8_t> vin, std::span<Float> vout) const;
    DFPBRT_CPU_GPU
    Float ToFloatLinear(Float v) const;
    DFPBRT_CPU_GPU
    void FromLinear(std::span<const Float> vin, std::span<uint8_t> vout) const;

    std::string ToString() const;

  private:
    Float gamma;
    std::array<Float, 256> applyLUT;
    std::array<Float, 1024> inverseLUT;
};

inline void ColorEncoding::ToLinear(std::span<const uint8_t> vin,
                                    std::span<Float> vout) const {
    auto tolin = [&](auto ptr) { return ptr->ToLinear(vin, vout); };
    Dispatch(tolin);
}

inline Float ColorEncoding::ToFloatLinear(Float v) const {
    auto tfl = [&](auto ptr) { return ptr->ToFloatLinear(v); };
    return Dispatch(tfl);
}

inline void ColorEncoding::FromLinear(std::span<const Float> vin,
                                      std::span<uint8_t> vout) const {
    auto fl = [&](auto ptr) { return ptr->FromLinear(vin, vout); };
    Dispatch(fl);
}

DFPBRT_CPU_GPU
inline Float LinearToSRGB(Float value) {
    if (value <= 0.0031308f)
        return 12.92f * value;
    // Minimax polynomial approximation from enoki's color.h.
    float sqrtValue = SafeSqrt(value);
    float p = EvaluatePolynomial(sqrtValue, -0.0016829072605308378f, 0.03453868659826638f,
                                 0.7642611304733891f, 2.0041169284241644f,
                                 0.7551545191665577f, -0.016202083165206348f);
    float q = EvaluatePolynomial(sqrtValue, 4.178892964897981e-7f,
                                 -0.00004375359692957097f, 0.03467195408529984f,
                                 0.6085338522168684f, 1.8970238036421054f, 1.f);
    return p / q * value;
}

DFPBRT_CPU_GPU
inline uint8_t LinearToSRGB8(Float value, Float dither = 0) {
    if (value <= 0)
        return 0;
    if (value >= 1)
        return 255;
    return Clamp(std::round(255.f * LinearToSRGB(value) + dither), 0, 255);
}

DFPBRT_CPU_GPU
inline Float SRGBToLinear(float value) {
    if (value <= 0.04045f)
        return value * (1 / 12.92f);
    // Minimax polynomial approximation from enoki's color.h.
    float p = EvaluatePolynomial(value, -0.0163933279112946f, -0.7386328024653209f,
                                 -11.199318357635072f, -47.46726633009393f,
                                 -36.04572663838034f);
    float q = EvaluatePolynomial(value, -0.004261480793199332f, -19.140923959601675f,
                                 -59.096406619244426f, -18.225745396846637f, 1.f);
    return p / q * value;
}

extern DFPBRT_CONST Float SRGBToLinearLUT[256];

DFPBRT_CPU_GPU
inline Float SRGB8ToLinear(uint8_t value) {
    return SRGBToLinearLUT[value];
}

// White Balance Definitions
// clang-format off
// These are the Bradford transformation matrices.
const SquareMatrix<3> LMSFromXYZ( 0.8951,  0.2664, -0.1614,
                                 -0.7502,  1.7135,  0.0367,
                                  0.0389, -0.0685,  1.0296);
const SquareMatrix<3> XYZFromLMS( 0.986993,   -0.147054,  0.159963,
                                  0.432305,    0.51836,   0.0492912,
                                 -0.00852866,  0.0400428, 0.968487);
// clang-format on

inline SquareMatrix<3> WhiteBalance(Point2f srcWhite, Point2f targetWhite) {
    // Find LMS coefficients for source and target white
    XYZ srcXYZ = XYZ::FromxyY(srcWhite), dstXYZ = XYZ::FromxyY(targetWhite);
    auto srcLMS = LMSFromXYZ * srcXYZ, dstLMS = LMSFromXYZ * dstXYZ;

    // Return white balancing matrix for source and target white
    SquareMatrix<3> LMScorrect = SquareMatrix<3>::Diag(
        dstLMS[0] / srcLMS[0], dstLMS[1] / srcLMS[1], dstLMS[2] / srcLMS[2]);
    return XYZFromLMS * LMScorrect * LMSFromXYZ;
}





}

#endif