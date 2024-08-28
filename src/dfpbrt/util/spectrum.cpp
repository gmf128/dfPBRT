#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/spectrum.h>
#include <dfpbrt/util/file.h>


// I don't know how this is happening (somehow via wingdi.h?), but not cool,
// Windows, not cool...
#ifdef RGB
#undef RGB
#endif

namespace dfpbrt{

XYZ SpectrumToXYZ(Spectrum s) {
    return XYZ(InnerProduct(&Spectra::X(), s), InnerProduct(&Spectra::Y(), s),
               InnerProduct(&Spectra::Z(), s)) /
           CIE_Y_integral;
}

std::string Spectrum::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto tostr = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(tostr);
}

// Spectrum Method Definitions
Float PiecewiseLinearSpectrum::operator()(Float lambda) const {
    // Handle _PiecewiseLinearSpectrum_ corner cases
    if (lambdas.empty() || lambda < lambdas.front() || lambda > lambdas.back())
        return 0;

    // Find offset to largest _lambdas_ below _lambda_ and interpolate
    int o = FindInterval(lambdas.size(), [&](int i) { return lambdas[i] <= lambda; });
    DCHECK(lambda >= lambdas[o] && lambda <= lambdas[o + 1]);
    Float t = (lambda - lambdas[o]) / (lambdas[o + 1] - lambdas[o]);
    return Lerp(t, values[o], values[o + 1]);
}

Float PiecewiseLinearSpectrum::MaxValue() const {
    if (values.empty())
        return 0;
    return *std::max_element(values.begin(), values.end());
}

PiecewiseLinearSpectrum::PiecewiseLinearSpectrum(std::span<const Float> l,
                                                 std::span<const Float> v,
                                                 Allocator alloc)
    : lambdas(l.begin(), l.end(), alloc), values(v.begin(), v.end(), alloc) {
    CHECK_EQ(lambdas.size(), values.size());
    for (size_t i = 0; i < lambdas.size() - 1; ++i)
        CHECK_LT(lambdas[i], lambdas[i + 1]);
}

std::string PiecewiseLinearSpectrum::ToString() const {
    std::string name = FindMatchingNamedSpectrum(this);
    if (!name.empty())
        return std::format("\"{0:s}\"", name);

    std::string ret = "[ PiecewiseLinearSpectrum ";
    for (size_t i = 0; i < lambdas.size(); ++i)
        ret += std::format("{} {} ", lambdas[i], values[i]);
    return ret + " ]";
}

std::optional<Spectrum> PiecewiseLinearSpectrum::Read(const std::string &fn,
                                                       Allocator alloc) {
    std::vector<Float> vals = ReadFloatFile(fn);
    if (vals.empty()) {
        // Warning("%s: unable to read spectrum file.", fn);
        return {};
    } else {
        if (vals.size() % 2 != 0) {
            // Warning("%s: extra value found in spectrum file.", fn);
            return {};
        }
        std::vector<Float> lambda, v;
        for (size_t i = 0; i < vals.size() / 2; ++i) {
            if (i > 0 && vals[2 * i] <= lambda.back()) {
                // Warning("%s: spectrum file invalid: at %d'th entry, "
                //         "wavelengths aren't "
                //         "increasing: %f >= %f.",
                //         fn, int(i), lambda.back(), vals[2 * i]);
                return {};
            }
            lambda.push_back(vals[2 * i]);
            v.push_back(vals[2 * i + 1]);
        }
        return Spectrum(alloc.new_object<PiecewiseLinearSpectrum>(lambda, v, alloc));
    }
}

PiecewiseLinearSpectrum *PiecewiseLinearSpectrum::FromInterleaved(
    std::span<const Float> samples, bool normalize, Allocator alloc) {
    CHECK_EQ(0, samples.size() % 2);
    int n = samples.size() / 2;
    std::vector<Float> lambda, v;

    // Extend samples to cover range of visible wavelengths if needed.
    if (samples[0] > Lambda_min) {
        lambda.push_back(Lambda_min - 1);
        v.push_back(samples[1]);
    }
    for (size_t i = 0; i < n; ++i) {
        lambda.push_back(samples[2 * i]);
        v.push_back(samples[2 * i + 1]);
        if (i > 0)
            CHECK_GT(lambda.back(), lambda[lambda.size() - 2]);
    }
    if (lambda.back() < Lambda_max) {
        lambda.push_back(Lambda_max + 1);
        v.push_back(v.back());
    }

    PiecewiseLinearSpectrum *spec =
        alloc.new_object<dfpbrt::PiecewiseLinearSpectrum>(lambda, v, alloc);

    if (normalize)
        // Normalize to have luminance of 1.
        spec->Scale(CIE_Y_integral / InnerProduct(spec, &Spectra::Y()));

    return spec;
}

std::string BlackbodySpectrum::ToString() const {
    return std::format("[ BlackbodySpectrum T: {} ]", T);
}

SampledSpectrum ConstantSpectrum::Sample(const SampledWavelengths &) const {
    return SampledSpectrum(c);
}

std::string ConstantSpectrum::ToString() const {
    return std::format("[ ConstantSpectrum c: {} ]", c);
}

std::string DenselySampledSpectrum::ToString() const {
    std::string s = std::format("[ DenselySampledSpectrum lambda_min: {} lambda_max: {} "
                                 "values: [ ",
                                 lambda_min, lambda_max);
    for (int i = 0; i < values.size(); ++i)
        s += std::string("{} ", values[i]);
    s += "] ]";
    return s;
}

std::string SampledSpectrum::ToString() const {
    std::string str = "[ ";
    for (int i = 0; i < NSpectrumSamples; ++i) {
        str += std::string("{}", values[i]);
        if (i + 1 < NSpectrumSamples)
            str += ", ";
    }
    str += " ]";
    return str;
}

std::string SampledWavelengths::ToString() const {
    std::string r = "[ SampledWavelengths lambda: [";
    for (size_t i = 0; i < lambda.size(); ++i)
        r += std::string(" {}{}", lambda[i], i != lambda.size() - 1 ? ',' : ' ');
    r += "] pdf: [";
    for (size_t i = 0; i < lambda.size(); ++i)
        r += std::string(" {}{}", pdf[i], i != pdf.size() - 1 ? ',' : ' ');
    r += "] ]";
    return r;
}

XYZ SampledSpectrum::ToXYZ(const SampledWavelengths &lambda) const {
    // Sample the $X$, $Y$, and $Z$ matching curves at _lambda_
    SampledSpectrum X = Spectra::X().Sample(lambda);
    SampledSpectrum Y = Spectra::Y().Sample(lambda);
    SampledSpectrum Z = Spectra::Z().Sample(lambda);

    // Evaluate estimator to compute $(x,y,z)$ coefficients
    SampledSpectrum pdf = lambda.PDF();
    return XYZ(SafeDiv(X * *this, pdf).Average(), SafeDiv(Y * *this, pdf).Average(),
               SafeDiv(Z * *this, pdf).Average()) /
           CIE_Y_integral;
}

Float SampledSpectrum::y(const SampledWavelengths &lambda) const {
    SampledSpectrum Ys = Spectra::Y().Sample(lambda);
    SampledSpectrum pdf = lambda.PDF();
    return SafeDiv(Ys * *this, pdf).Average() / CIE_Y_integral;
}


}