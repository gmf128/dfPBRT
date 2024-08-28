#ifndef DFPBRT_UTIL_SPECTRUM_H
#define DFPBRT_UTIL_SPECTRUM_H

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/math.h>
#include <dfpbrt/util/vecmath.h>
#include <dfpbrt/util/sampling.h>
#include <dfpbrt/util/taggedptr.h>
#include <dfpbrt/util/color.h>


namespace dfpbrt{
// The range of visible range
constexpr Float Lambda_min = 360, Lambda_max = 830;

static constexpr int NSpectrumSamples = 4;

static constexpr Float CIE_Y_integral = 106.856895;

// Spectrum Definition
// all possible spectrum types, they are all derived class of the base class Spectrum
class BlackbodySpectrum;
class ConstantSpectrum;
class PiecewiseLinearSpectrum;
class DenselySampledSpectrum;
class RGBAlbedoSpectrum;
class RGBUnboundedSpectrum;
class RGBIlluminantSpectrum;


// The spectrum class is only an interface, i.e. the functions must be implemented by all of child classes.
class Spectrum: public TaggedPointer <BlackbodySpectrum, ConstantSpectrum, PiecewiseLinearSpectrum,
                                DenselySampledSpectrum, RGBAlbedoSpectrum, RGBUnboundedSpectrum, 
                                RGBIlluminantSpectrum>{
public:
    // Spectrum Interface
    // same constructor functions
    using TaggedPointer::TaggedPointer;
    std::string ToString() const;

    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const;

    DFPBRT_CPU_GPU
    Float MaxValue() const;

    DFPBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;

};

// Spectrum Function Declarations

// Ideal blackbody
DFPBRT_CPU_GPU inline Float Blackbody(Float lambda, Float T) {
    if (T <= 0)
        return 0;
    const Float c = 299792458.f;
    const Float h = 6.62606957e-34f;
    const Float kb = 1.3806488e-23f;
    // Return emitted radiance for blackbody at wavelength _lambda_
    Float l = lambda * 1e-9f;
    Float Le = (2 * h * c * c) / (Pow<5>(l) * (FastExp((h * c) / (l * kb * T)) - 1));
    CHECK(!IsNaN(Le));
    return Le;
}

namespace Spectra {
DenselySampledSpectrum D(Float T, Allocator alloc);
}  // namespace Spectra

Float SpectrumToPhotometric(Spectrum s);

XYZ SpectrumToXYZ(Spectrum s);

// SampledSpectrum Definition
class SampledSpectrum {
  public:
    // Constructors
    SampledSpectrum() = default;
    DFPBRT_CPU_GPU
    explicit SampledSpectrum(Float c) { values.fill(c); }
    DFPBRT_CPU_GPU
    SampledSpectrum(std::span<const Float> v) {
        DCHECK_EQ(NSpectrumSamples, v.size());
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] = v[i];
    }
    // SampledSpectrum Public Methods
    DFPBRT_CPU_GPU
    SampledSpectrum operator+(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret += s;
    }

    DFPBRT_CPU_GPU
    SampledSpectrum &operator-=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] -= s.values[i];
        return *this;
    }
    DFPBRT_CPU_GPU
    SampledSpectrum operator-(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret -= s;
    }
    DFPBRT_CPU_GPU
    friend SampledSpectrum operator-(Float a, const SampledSpectrum &s) {
        DCHECK(!IsNaN(a));
        SampledSpectrum ret;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.values[i] = a - s.values[i];
        return ret;
    }

    DFPBRT_CPU_GPU
    SampledSpectrum &operator*=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] *= s.values[i];
        return *this;
    }
    DFPBRT_CPU_GPU
    SampledSpectrum operator*(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret *= s;
    }
    DFPBRT_CPU_GPU
    SampledSpectrum operator*(Float a) const {
        DCHECK(!IsNaN(a));
        SampledSpectrum ret = *this;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.values[i] *= a;
        return ret;
    }
    DFPBRT_CPU_GPU
    SampledSpectrum &operator*=(Float a) {
        DCHECK(!IsNaN(a));
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] *= a;
        return *this;
    }
    DFPBRT_CPU_GPU
    friend SampledSpectrum operator*(Float a, const SampledSpectrum &s) { return s * a; }

    DFPBRT_CPU_GPU
    SampledSpectrum &operator/=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i) {
            DCHECK_NE(0, s.values[i]);
            values[i] /= s.values[i];
        }
        return *this;
    }
    DFPBRT_CPU_GPU
    SampledSpectrum operator/(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret /= s;
    }
    DFPBRT_CPU_GPU
    SampledSpectrum &operator/=(Float a) {
        DCHECK_NE(a, 0);
        DCHECK(!IsNaN(a));
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] /= a;
        return *this;
    }
    DFPBRT_CPU_GPU
    SampledSpectrum operator/(Float a) const {
        SampledSpectrum ret = *this;
        return ret /= a;
    }

    DFPBRT_CPU_GPU
    SampledSpectrum operator-() const {
        SampledSpectrum ret;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.values[i] = -values[i];
        return ret;
    }
    DFPBRT_CPU_GPU
    bool operator==(const SampledSpectrum &s) const { return values == s.values; }
    DFPBRT_CPU_GPU
    bool operator!=(const SampledSpectrum &s) const { return values != s.values; }

    std::string ToString() const;

    DFPBRT_CPU_GPU
    bool HasNaNs() const {
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (IsNaN(values[i]))
                return true;
        return false;
    }

    DFPBRT_CPU_GPU
    XYZ ToXYZ(const SampledWavelengths &lambda) const;
    DFPBRT_CPU_GPU
    RGB ToRGB(const SampledWavelengths &lambda, const RGBColorSpace &cs) const;
    DFPBRT_CPU_GPU
    Float y(const SampledWavelengths &lambda) const;

    

    DFPBRT_CPU_GPU
    Float operator[](int i) const {
        DCHECK(i >= 0 && i < NSpectrumSamples);
        return values[i];
    }
    DFPBRT_CPU_GPU
    Float &operator[](int i) {
        DCHECK(i >= 0 && i < NSpectrumSamples);
        return values[i];
    }

    DFPBRT_CPU_GPU
    explicit operator bool() const {
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (values[i] != 0)
                return true;
        return false;
    }

    DFPBRT_CPU_GPU
    SampledSpectrum &operator+=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] += s.values[i];
        return *this;
    }

    DFPBRT_CPU_GPU
    Float MinComponentValue() const {
        Float m = values[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            m = std::min(m, values[i]);
        return m;
    }
    DFPBRT_CPU_GPU
    Float MaxComponentValue() const {
        Float m = values[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            m = std::max(m, values[i]);
        return m;
    }
    DFPBRT_CPU_GPU
    Float Average() const {
        Float sum = values[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            sum += values[i];
        return sum / NSpectrumSamples;
    }

  private:
    std::array<Float, NSpectrumSamples> values;
};

// SampledWavelengths Definitions
class SampledWavelengths {
  public:
    // SampledWavelengths Public Methods
    DFPBRT_CPU_GPU
    bool operator==(const SampledWavelengths &swl) const {
        return lambda == swl.lambda && pdf == swl.pdf;
    }
    DFPBRT_CPU_GPU
    bool operator!=(const SampledWavelengths &swl) const {
        return lambda != swl.lambda || pdf != swl.pdf;
    }

    std::string ToString() const;

    DFPBRT_CPU_GPU
    static SampledWavelengths SampleUniform(Float u, Float lambda_min = Lambda_min,
                                            Float lambda_max = Lambda_max) {
        SampledWavelengths swl;
        // Sample first wavelength using _u_, u is a random number in [0, 1]
        swl.lambda[0] = Lerp(u, lambda_min, lambda_max);

        // Initialize _lambda_ for remaining wavelengths
        Float delta = (lambda_max - lambda_min) / NSpectrumSamples;
        for (int i = 1; i < NSpectrumSamples; ++i) {
            swl.lambda[i] = swl.lambda[i - 1] + delta;
            if (swl.lambda[i] > lambda_max)
                swl.lambda[i] = lambda_min + (swl.lambda[i] - lambda_max);
        }

        // Compute PDF for sampled wavelengths
        for (int i = 0; i < NSpectrumSamples; ++i)
            swl.pdf[i] = 1 / (lambda_max - lambda_min);// uniform

        return swl;
    }

    DFPBRT_CPU_GPU
    Float operator[](int i) const { return lambda[i]; }
    DFPBRT_CPU_GPU
    Float &operator[](int i) { return lambda[i]; }
    // It is not surprising to use SampledSpectrum to store pdf
    DFPBRT_CPU_GPU
    SampledSpectrum PDF() const { return SampledSpectrum(pdf); }

    /**
     * In some cases, different wavelengths of light may follow different paths after a scattering event. 
     * The most common example is when light undergoes dispersion and different wavelengths of light refract to different directions. When this happens, it is no longer possible to track multiple wavelengths of light with a single ray. 
     * For this case, SampledWavelengths provides the capability of terminating all but one of the wavelengths; subsequent computations can then consider the single surviving wavelength exclusively.
     */
    DFPBRT_CPU_GPU
    void TerminateSecondary() {
        if (SecondaryTerminated())
            return;
        // Update wavelength probabilities for termination
        /**
         * The wavelength stored in lambda[0] is always the survivor: 
         * there is no need to randomly select the surviving wavelength so long as each lambda value was randomly sampled from the same distribution as is the case with SampleUniform(), 
         * for example. Note that this means that it would be incorrect for SampledWavelengths::SampleUniform() to always place lambda[0] in a first wavelength stratum between lambda_min and lambda_min+delta, lambda[1] in the second, and so forth
         */
        for (int i = 1; i < NSpectrumSamples; ++i)
        /**
         * Terminated wavelengths have their PDF values set to zero; 
         * code that computes Monte Carlo estimates using SampledWavelengths must therefore detect this case and ignore terminated wavelengths accordingly. 
         * The surviving wavelength’s PDF is updated to account for the termination event by multiplying it by the probability of a wavelength surviving termination, 1 / NSpectrumSamples. 
         * (This is similar to how applying Russian roulette affects the Monte Carlo estimator—see Section 2.2.4.)
         */
            pdf[i] = 0;
        pdf[0] /= NSpectrumSamples;
    }

    DFPBRT_CPU_GPU
    bool SecondaryTerminated() const {
        for (int i = 1; i < NSpectrumSamples; ++i)
            if (pdf[i] != 0)
                return false;
        return true;
    }

    DFPBRT_CPU_GPU
    static SampledWavelengths SampleVisible(Float u) {
        //Here u is also a random variable range[0, 1]
        SampledWavelengths swl;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            // Compute _up_ for $i$th wavelength sample
            Float up = u + (Float(i) / NSpectrumSamples);
            if (up > 1)
                up -= 1;

            swl.lambda[i] = SampleVisibleWavelengths(up);
            swl.pdf[i] = VisibleWavelengthsPDF(swl.lambda[i]);
        }
        return swl;
    }

  private:
    // SampledWavelengths Private Members
    std::array<Float, NSpectrumSamples> lambda, pdf;
};

// Spectrum Definitions

//The simplest is constant spectrum
class ConstantSpectrum {
    //Note that it does not inherit from Spectrum. This is another difference from using traditional C++ abstract base classes with virtual functions—
    //as far as the C++ type system is concerned, there is no explicit connection between ConstantSpectrum and Spectrum.
  public:
    DFPBRT_CPU_GPU
    ConstantSpectrum(Float c) : c(c) {}
    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const { return c; }
    // ConstantSpectrum Public Methods
    DFPBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &) const;

    DFPBRT_CPU_GPU
    Float MaxValue() const { return c; }

    std::string ToString() const;

  private:
    Float c;
};

//DenselySampledSpectrum stores a spectral distribution sampled at 1 nm intervals over a given range of integer wavelengths
class DenselySampledSpectrum {
  public:
    // DenselySampledSpectrum Public Methods
    DenselySampledSpectrum(int lambda_min = Lambda_min, int lambda_max = Lambda_max,
                           Allocator alloc = {})
        : lambda_min(lambda_min),
          lambda_max(lambda_max),
          //values initialization: values(length, allocator)
          values((size_t)(lambda_max - lambda_min + 1), alloc) {}
    DenselySampledSpectrum(Spectrum s, Allocator alloc)
        : DenselySampledSpectrum(s, Lambda_min, Lambda_max, alloc) {}
    DenselySampledSpectrum(const DenselySampledSpectrum &s, Allocator alloc)
        : lambda_min(s.lambda_min),
          lambda_max(s.lambda_max),
          values(s.values.begin(), s.values.end(), alloc) {}

    // 
    //Random number --> SampledWavelengths ------->Spectrum.Sample() ----> SampledSpectrum
    //
    DFPBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            int offset = std::lround(lambda[i]) - lambda_min;
            if (offset < 0 || offset >= values.size())
                s[i] = 0;
            else
                s[i] = values[offset];
        }
        return s;
    }

    DFPBRT_CPU_GPU
    void Scale(Float s) {
        // Scale the function value ,not changing the range(lambda_min, lambda_max)
        for (Float &v : values)
            v *= s;
    }

    DFPBRT_CPU_GPU
    Float MaxValue() const { return *std::max_element(values.begin(), values.end()); }

    std::string ToString() const;

    DenselySampledSpectrum(Spectrum spec, int lambda_min = Lambda_min,
                           int lambda_max = Lambda_max, Allocator alloc = {})
        : lambda_min(lambda_min),
          lambda_max(lambda_max),
          values(lambda_max - lambda_min + 1, alloc) {
        CHECK_GE(lambda_max, lambda_min);
        if (spec) //spectrum has boolean conversion? explicit TaggedPointer::operator bool() const { return (bits & ptrMask) != 0; }
            for (int lambda = lambda_min; lambda <= lambda_max; ++lambda)
                values[lambda - lambda_min] = spec(lambda);
    }
    // A static function. Sample a function in densly-sampled way into a DenslySampledSpectrum object
    template <typename F>
    static DenselySampledSpectrum SampleFunction(F func, int lambda_min = Lambda_min,
                                                 int lambda_max = Lambda_max,
                                                 Allocator alloc = {}) {
        DenselySampledSpectrum s(lambda_min, lambda_max, alloc);
        for (int lambda = lambda_min; lambda <= lambda_max; ++lambda)
            s.values[lambda - lambda_min] = func(lambda);
        return s;
    }

    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const {
        DCHECK_GT(lambda, 0);
        int offset = std::lround(lambda) - lambda_min;
        if (offset < 0 || offset >= values.size())
            return 0;
        return values[offset];
    }

    DFPBRT_CPU_GPU
    bool operator==(const DenselySampledSpectrum &d) const {
        if (lambda_min != d.lambda_min || lambda_max != d.lambda_max ||
            values.size() != d.values.size())
            return false;
        for (size_t i = 0; i < values.size(); ++i)
            if (values[i] != d.values[i])
                return false;
        return true;
    }

  private:
    friend struct std::hash<dfpbrt::DenselySampledSpectrum>;
    // DenselySampledSpectrum Private Members
    int lambda_min, lambda_max;
    std::vector<Float, Allocator> values;
};

class PiecewiseLinearSpectrum {
  public:
    // PiecewiseLinearSpectrum Public Methods
    PiecewiseLinearSpectrum() = default;

    DFPBRT_CPU_GPU
    void Scale(Float s) {
        for (Float &v : values)
            v *= s;
    }

    DFPBRT_CPU_GPU
    Float MaxValue() const;

    DFPBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = (*this)(lambda[i]);
        return s;
    }
    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const;

    std::string ToString() const;

    PiecewiseLinearSpectrum(std::span<const Float> lambdas,
                            std::span<const Float> values, Allocator alloc = {});

    static std::optional<Spectrum> Read(const std::string &filename, Allocator alloc);

    static PiecewiseLinearSpectrum *FromInterleaved(std::span<const Float> samples,
                                                    bool normalize, Allocator alloc);

  private:
    // PiecewiseLinearSpectrum Private Members
    std::vector<Float, Allocator> lambdas, values;
};

class BlackbodySpectrum {
  public:
    // BlackbodySpectrum Public Methods
    DFPBRT_CPU_GPU
    BlackbodySpectrum(Float T) : T(T) {
        // Compute blackbody normalization constant for given temperature
        Float lambdaMax = 2.8977721e-3f / T;
        normalizationFactor = 1 / Blackbody(lambdaMax * 1e9f, T);
    }

    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const {
        return Blackbody(lambda, T) * normalizationFactor;
    }

    DFPBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = Blackbody(lambda[i], T) * normalizationFactor;
        return s;
    }

    DFPBRT_CPU_GPU
    Float MaxValue() const { return 1.f; }

    std::string ToString() const;

  private:
    // BlackbodySpectrum Private Members
    Float T;
    Float normalizationFactor;
};

class RGBAlbedoSpectrum {
  public:
    // RGBAlbedoSpectrum Public Methods
    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const { return rsp(lambda); }
    DFPBRT_CPU_GPU
    Float MaxValue() const { return rsp.MaxValue(); }

    DFPBRT_CPU_GPU
    RGBAlbedoSpectrum(const RGBColorSpace &cs, RGB rgb);

    DFPBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = rsp(lambda[i]);
        return s;
    }

    std::string ToString() const;

  private:
    // RGBAlbedoSpectrum Private Members
    RGBSigmoidPolynomial rsp;
};

class RGBUnboundedSpectrum {
  public:
    // RGBUnboundedSpectrum Public Methods
    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const { return scale * rsp(lambda); }
    DFPBRT_CPU_GPU
    Float MaxValue() const { return scale * rsp.MaxValue(); }

    DFPBRT_CPU_GPU
    RGBUnboundedSpectrum(const RGBColorSpace &cs, RGB rgb);

    DFPBRT_CPU_GPU
    RGBUnboundedSpectrum() : rsp(0, 0, 0), scale(0) {}

    DFPBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = scale * rsp(lambda[i]);
        return s;
    }

    std::string ToString() const;

  private:
    // RGBUnboundedSpectrum Private Members
    Float scale = 1;
    RGBSigmoidPolynomial rsp;
};

class RGBIlluminantSpectrum {
  public:
    // RGBIlluminantSpectrum Public Methods
    RGBIlluminantSpectrum() = default;
    DFPBRT_CPU_GPU
    RGBIlluminantSpectrum(const RGBColorSpace &cs, RGB rgb);

    DFPBRT_CPU_GPU
    Float operator()(Float lambda) const {
        if (!illuminant)
            return 0;
        return scale * rsp(lambda) * (*illuminant)(lambda);
    }

    DFPBRT_CPU_GPU
    Float MaxValue() const {
        if (!illuminant)
            return 0;
        return scale * rsp.MaxValue() * illuminant->MaxValue();
    }

    DFPBRT_CPU_GPU
    const DenselySampledSpectrum *Illuminant() const { return illuminant; }

    DFPBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        if (!illuminant)
            return SampledSpectrum(0);
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = scale * rsp(lambda[i]);
        return s * illuminant->Sample(lambda);
    }

    std::string ToString() const;

  private:
    // RGBIlluminantSpectrum Private Members
    Float scale;
    RGBSigmoidPolynomial rsp;
    const DenselySampledSpectrum *illuminant;
};

// SampledSpectrum Inline Functions
DFPBRT_CPU_GPU inline SampledSpectrum SafeDiv(SampledSpectrum a, SampledSpectrum b) {
    SampledSpectrum r;
    for (int i = 0; i < NSpectrumSamples; ++i)
        r[i] = (b[i] != 0) ? a[i] / b[i] : 0.;
    return r;
}

template <typename U, typename V>
DFPBRT_CPU_GPU inline SampledSpectrum Clamp(const SampledSpectrum &s, U low, V high) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = dfpbrt::Clamp(s[i], low, high);
    DCHECK(!ret.HasNaNs());
    return ret;
}

// Clamp to positive
DFPBRT_CPU_GPU
inline SampledSpectrum ClampZero(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::max<Float>(0, s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

DFPBRT_CPU_GPU
inline SampledSpectrum Sqrt(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::sqrt(s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

DFPBRT_CPU_GPU
inline SampledSpectrum SafeSqrt(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = SafeSqrt(s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

DFPBRT_CPU_GPU
inline SampledSpectrum Pow(const SampledSpectrum &s, Float e) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::pow(s[i], e);
    return ret;
}

DFPBRT_CPU_GPU
inline SampledSpectrum Exp(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::exp(s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

DFPBRT_CPU_GPU
inline SampledSpectrum FastExp(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = FastExp(s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

DFPBRT_CPU_GPU
inline SampledSpectrum Bilerp(std::array<Float, 2> p,
                              std::span<const SampledSpectrum> v) {
    return ((1 - p[0]) * (1 - p[1]) * v[0] + p[0] * (1 - p[1]) * v[1] +
            (1 - p[0]) * p[1] * v[2] + p[0] * p[1] * v[3]);
}

DFPBRT_CPU_GPU
inline SampledSpectrum Lerp(Float t, const SampledSpectrum &s1,
                            const SampledSpectrum &s2) {
    return (1 - t) * s1 + t * s2;
}

// Spectral Data Declarations
namespace Spectra {

void Init(Allocator alloc);

DFPBRT_CPU_GPU
inline const DenselySampledSpectrum &X() {
#ifdef DFPBRT_IS_GPU_CODE
    extern DFPBRT_GPU DenselySampledSpectrum *xGPU;
    return *xGPU;
#else
    extern DenselySampledSpectrum *x;
    return *x;
#endif
}

DFPBRT_CPU_GPU
inline const DenselySampledSpectrum &Y() {
#ifdef DFPBRT_IS_GPU_CODE
    extern DFPBRT_GPU DenselySampledSpectrum *yGPU;
    return *yGPU;
#else
    extern DenselySampledSpectrum *y;
    return *y;
#endif
}

DFPBRT_CPU_GPU
inline const DenselySampledSpectrum &Z() {
#ifdef DFPBRT_IS_GPU_CODE
    extern DFPBRT_GPU DenselySampledSpectrum *zGPU;
    return *zGPU;
#else
    extern DenselySampledSpectrum *z;
    return *z;
#endif
}

}  // namespace Spectra

// Spectral Function Declarations
Spectrum GetNamedSpectrum(std::string name);

std::string FindMatchingNamedSpectrum(Spectrum s);

namespace Spectra {
inline const DenselySampledSpectrum &X();
inline const DenselySampledSpectrum &Y();
inline const DenselySampledSpectrum &Z();
}  // namespace Spectra

// Spectrum Inline Functions
inline Float InnerProduct(Spectrum f, Spectrum g) {
    Float integral = 0;
    for (Float lambda = Lambda_min; lambda <= Lambda_max; ++lambda)
        integral += f(lambda) * g(lambda);
    return integral;
}

// Spectrum Inline Method Definitions
inline Float Spectrum::operator()(Float lambda) const {
    auto op = [&](auto ptr) { return (*ptr)(lambda); };
    return Dispatch(op);
}

inline SampledSpectrum Spectrum::Sample(const SampledWavelengths &lambda) const {
    auto samp = [&](auto ptr) { return ptr->Sample(lambda); };
    return Dispatch(samp);
}

inline Float Spectrum::MaxValue() const {
    auto max = [&](auto ptr) { return ptr->MaxValue(); };
    return Dispatch(max);

}

}


#endif