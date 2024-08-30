#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/color.h>
#include <dfpbrt/util/colorspace.h>

#include <ext/gtest/gtest.h>

using namespace dfpbrt;

TEST(RGBColorSpace, XYZIsOthognal) {
    Spectrum X(&Spectra::X());
    Spectrum Y(&Spectra::Y());
    Spectrum Z(&Spectra::Z());
    Float XX = InnerProduct(X, X);
    Float XY = InnerProduct(X, Y);
    Float XZ = InnerProduct(X, Z);
    Float YY = InnerProduct(Y, Y);
    Float YZ = InnerProduct(Y, Z);
    Float ZZ = InnerProduct(Z, Z);
    std::cout << "XX: " << XX << std::endl;
    std::cout << "XY: " << XY << std::endl;
    std::cout << "XZ: " << XZ << std::endl;
    std::cout << "YY: " << YY << std::endl;
    std::cout << "YZ: " << YZ << std::endl;
    std::cout << "ZZ: " << ZZ << std::endl;
}

TEST(RGBColorSpace, RGBXYZ) {
    for (const RGBColorSpace &cs :
         {*RGBColorSpace::ACES2065_1, *RGBColorSpace::Rec2020, *RGBColorSpace::sRGB}) {
        //std::cout << cs.XYZFromRGB.ToString();
        XYZ xyz = cs.ToXYZ({1, 1, 1});
        RGB rgb = cs.ToRGB(xyz);
        Spectrum illuminant (&cs.illuminant);
        XYZ W = SpectrumToXYZ(illuminant);
        RGB w = cs.ToRGB(W);
        EXPECT_LT(std::abs(1 - rgb[0]), 1e-4);
        EXPECT_LT(std::abs(1 - rgb[1]), 1e-4);
        EXPECT_LT(std::abs(1 - rgb[2]), 1e-4);
        EXPECT_LT(std::abs(1 - w[0]), 1e-4);
        EXPECT_LT(std::abs(1 - w[1]), 1e-4);
        EXPECT_LT(std::abs(1 - w[2]), 1e-4);
    }
}

TEST(RGBColorSpace, sRGB) {
    const RGBColorSpace &sRGB = *RGBColorSpace::sRGB;
    std::cout << sRGB.RGBFromXYZ.ToString();
    // Make sure the matrix values are sensible by throwing the x, y, and z
    // basis vectors at it to pull out columns.
    // Right answer:
    // 3.2404542 -1.5371385 -0.4985314
    // -0.9692660  1.8760108  0.0415560
    // 0.0556434 -0.2040259  1.0572252
    RGB rgb = sRGB.ToRGB({1, 0, 0});
    std::cout << rgb.ToString() << std::endl;
    EXPECT_LT(std::abs(3.2406 - rgb[0]), 1e-3);
    EXPECT_LT(std::abs(-.9689 - rgb[1]), 1e-3);
    EXPECT_LT(std::abs(.0557 - rgb[2]), 1e-3);

    rgb = sRGB.ToRGB({0, 1, 0});
    std::cout << rgb.ToString() << std::endl;
    EXPECT_LT(std::abs(-1.5372 - rgb[0]), 1e-3);
    EXPECT_LT(std::abs(1.8758 - rgb[1]), 1e-3);
    EXPECT_LT(std::abs(-.2040 - rgb[2]), 1e-3);

    rgb = sRGB.ToRGB({0, 0, 1});
    std::cout << rgb.ToString() << std::endl;
    EXPECT_LT(std::abs(-.4986 - rgb[0]), 1e-3);
    EXPECT_LT(std::abs(.0415 - rgb[1]), 1e-3);
    EXPECT_LT(std::abs(1.0570 - rgb[2]), 1e-3);
}

TEST(RGBColorSpace, StdIllumWhitesRGB) {
    XYZ xyz = SpectrumToXYZ(&RGBColorSpace::sRGB->illuminant);
    RGB rgb = RGBColorSpace::sRGB->ToRGB(xyz);
    EXPECT_GE(rgb.r, .99);
    EXPECT_LE(rgb.r, 1.01);
    EXPECT_GE(rgb.g, .99);
    EXPECT_LE(rgb.g, 1.01);
    EXPECT_GE(rgb.b, .99);
    EXPECT_LE(rgb.b, 1.01);
}

TEST(RGBColorSpace, StdIllumWhiteRec2020) {
    XYZ xyz = SpectrumToXYZ(&RGBColorSpace::Rec2020->illuminant);
    RGB rgb = RGBColorSpace::Rec2020->ToRGB(xyz);
    EXPECT_GE(rgb.r, .99);
    EXPECT_LE(rgb.r, 1.01);
    EXPECT_GE(rgb.g, .99);
    EXPECT_LE(rgb.g, 1.01);
    EXPECT_GE(rgb.b, .99);
    EXPECT_LE(rgb.b, 1.01);
}

TEST(RGBColorSpace, StdIllumWhiteACES2065_1) {
    XYZ xyz = SpectrumToXYZ(&RGBColorSpace::ACES2065_1->illuminant);
    RGB rgb = RGBColorSpace::ACES2065_1->ToRGB(xyz);
    EXPECT_GE(rgb.r, .99);
    EXPECT_LE(rgb.r, 1.01);
    EXPECT_GE(rgb.g, .99);
    EXPECT_LE(rgb.g, 1.01);
    EXPECT_GE(rgb.b, .99);
    EXPECT_LE(rgb.b, 1.01);
}