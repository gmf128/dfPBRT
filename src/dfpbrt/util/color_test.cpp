#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/color.h>
#include <dfpbrt/util/colorspace.h>

#include <ext/gtest/gtest.h>

using namespace dfpbrt;

TEST(RGBColorSpace, RGBXYZ) {
    for (const RGBColorSpace &cs :
         {*RGBColorSpace::ACES2065_1, *RGBColorSpace::Rec2020, *RGBColorSpace::sRGB}) {
        XYZ xyz = cs.ToXYZ({1, 1, 1});
        RGB rgb = cs.ToRGB(xyz);
        EXPECT_LT(std::abs(1 - rgb[0]), 1e-4);
        EXPECT_LT(std::abs(1 - rgb[1]), 1e-4);
        EXPECT_LT(std::abs(1 - rgb[2]), 1e-4);
    }
}