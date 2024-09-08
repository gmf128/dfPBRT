#ifndef DFPBRT_FILM_H
#define DFPBRT_FILM_H

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/color.h>
#include <dfpbrt/util/colorspace.h>

namespace dfpbrt{

// PixelSensor Definition
class PixelSensor {
  public:
    // PixelSensor Public Methods
    static PixelSensor *Create(const ParameterDictionary &parameters,
                               const RGBColorSpace *colorSpace, Float exposureTime,
                               const FileLoc *loc, Allocator alloc);

    static PixelSensor *CreateDefault(Allocator alloc = {});

    PixelSensor(Spectrum r, Spectrum g, Spectrum b, const RGBColorSpace *outputColorSpace,
                Spectrum sensorIllum, Float imagingRatio, Allocator alloc)
        : r_bar(r, alloc), g_bar(g, alloc), b_bar(b, alloc), imagingRatio(imagingRatio) {
        // Compute XYZ from camera RGB matrix
        // Compute _rgbCamera_ values for training swatches
        Float rgbCamera[nSwatchReflectances][3];
        for (int i = 0; i < nSwatchReflectances; ++i) {
            RGB rgb = ProjectReflectance<RGB>(swatchReflectances[i], sensorIllum, &r_bar,
                                              &g_bar, &b_bar);
            for (int c = 0; c < 3; ++c)
                rgbCamera[i][c] = rgb[c];
        }

        // Compute _xyzOutput_ values for training swatches
        Float xyzOutput[24][3];
        Float sensorWhiteG = InnerProduct(sensorIllum, &g_bar);
        Float sensorWhiteY = InnerProduct(sensorIllum, &Spectra::Y());
        for (size_t i = 0; i < nSwatchReflectances; ++i) {
            Spectrum s = swatchReflectances[i];
            XYZ xyz =
                ProjectReflectance<XYZ>(s, &outputColorSpace->illuminant, &Spectra::X(),
                                        &Spectra::Y(), &Spectra::Z()) *
                (sensorWhiteY / sensorWhiteG);
            for (int c = 0; c < 3; ++c)
                xyzOutput[i][c] = xyz[c];
        }

        // Initialize _XYZFromSensorRGB_ using linear least squares
        pstd::optional<SquareMatrix<3>> m =
            LinearLeastSquares(rgbCamera, xyzOutput, nSwatchReflectances);
        if (!m)
            ErrorExit("Sensor XYZ from RGB matrix could not be solved.");
        XYZFromSensorRGB = *m;
    }

    PixelSensor(const RGBColorSpace *outputColorSpace, Spectrum sensorIllum,
                Float imagingRatio, Allocator alloc)
        : r_bar(&Spectra::X(), alloc),
          g_bar(&Spectra::Y(), alloc),
          b_bar(&Spectra::Z(), alloc),
          imagingRatio(imagingRatio) {
        // Compute white balancing matrix for XYZ _PixelSensor_
        if (sensorIllum) {
            Point2f sourceWhite = SpectrumToXYZ(sensorIllum).xy();
            Point2f targetWhite = outputColorSpace->w;
            XYZFromSensorRGB = WhiteBalance(sourceWhite, targetWhite);
        }
    }

    PBRT_CPU_GPU
    RGB ToSensorRGB(SampledSpectrum L, const SampledWavelengths &lambda) const {
        L = SafeDiv(L, lambda.PDF());
        return imagingRatio * RGB((r_bar.Sample(lambda) * L).Average(),
                                  (g_bar.Sample(lambda) * L).Average(),
                                  (b_bar.Sample(lambda) * L).Average());
    }

    // PixelSensor Public Members
    SquareMatrix<3> XYZFromSensorRGB;

  private:
    // PixelSensor Private Methods
    template <typename Triplet>
    static Triplet ProjectReflectance(Spectrum r, Spectrum illum, Spectrum b1,
                                      Spectrum b2, Spectrum b3);

    // PixelSensor Private Members
    DenselySampledSpectrum r_bar, g_bar, b_bar;
    Float imagingRatio;
    static constexpr int nSwatchReflectances = 24;
    static Spectrum swatchReflectances[nSwatchReflectances];
};

// PixelSensor Inline Methods
template <typename Triplet>
inline Triplet PixelSensor::ProjectReflectance(Spectrum refl, Spectrum illum, Spectrum b1,
                                               Spectrum b2, Spectrum b3) {
    Triplet result;
    Float g_integral = 0;
    for (Float lambda = Lambda_min; lambda <= Lambda_max; ++lambda) {
        g_integral += b2(lambda) * illum(lambda);
        result[0] += b1(lambda) * refl(lambda) * illum(lambda);
        result[1] += b2(lambda) * refl(lambda) * illum(lambda);
        result[2] += b3(lambda) * refl(lambda) * illum(lambda);
    }
    return result / g_integral;
}
}

#endif