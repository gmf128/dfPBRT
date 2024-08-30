#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/colorspace.h>

namespace dfpbrt{
    RGBColorSpace::RGBColorSpace(Point2f r, Point2f g, Point2f b, Spectrum illuminant,
                  const RGBToSpectrumTable *rgbToSpectrumTable, Allocator alloc):
                  r(r), g(g), b(b), illuminant(illuminant, alloc), rgbToSpectrumTable(rgbToSpectrumTable){
                    //TODO: REMAIN TO EXPLORE
        #if 0
            // Define an RGB color space using the chromaticities of r g b w(white point) in xyY space
            // w is the xyY coord of spectrum illuminant, first we will get its XYZ coords
            XYZ W = SpectrumToXYZ(illuminant);
            w = W.xy();
            // Then, we try to calculate XYZtoRGB matrix and its inverse
            // [x, y, z] = [\int RX, \int RY, \int RZ] * [r, g, b]
            //             [\int GX, \int GY, \int GZ]
            //             [\int BX, \int BY, \int BZ]
            // \int RX^prime = X(R) / Y(R)
            XYZ R_prime = XYZ::FromxyY(r);  // for now, Y is set to default one, the actual value of Y is remained to calculate
            XYZ G_prime = XYZ::FromxyY(g);
            XYZ B_prime = XYZ::FromxyY(b);

            Float Y_R = Float(1) / (InnerProduct(R_prime.X, W.X, R_prime.Y, W.Y, R_prime.Z, W.Z));
            Float Y_B = Float(1) / (InnerProduct(B_prime.X, W.X, B_prime.Y, W.Y, B_prime.Z, W.Z));
            Float Y_G = Float(1) / (InnerProduct(G_prime.X, W.X, G_prime.Y, W.Y, G_prime.Z, W.Z));

            XYZ R = R_prime * Y_R;
            XYZ G = G_prime * Y_G;
            XYZ B = B_prime * Y_B;

            RGBFromXYZ = SquareMatrix<3>(R.X, R.Y, R.Z, G.X, G.Y, G.Z, B.X, B.Y, B.Z);
            XYZFromRGB = InvertOrExit(RGBFromXYZ);
        #endif
            // Compute whitepoint primaries and XYZ coordinates
            XYZ W = SpectrumToXYZ(illuminant);
            w = W.xy();
            XYZ R = XYZ::FromxyY(r), G = XYZ::FromxyY(g), B = XYZ::FromxyY(b);

            // Initialize XYZ color space conversion matrices
            SquareMatrix<3> rgb(R.X, G.X, B.X, R.Y, G.Y, B.Y, R.Z, G.Z, B.Z);
            XYZ C = InvertOrExit(rgb) * W;
            XYZFromRGB = rgb * SquareMatrix<3>::Diag(C[0], C[1], C[2]);
            RGBFromXYZ = InvertOrExit(XYZFromRGB);
        }

    SquareMatrix<3> ConvertRGBColorSpace(const RGBColorSpace &from, const RGBColorSpace &to) {
        if (from == to)
            return {};
        return to.RGBFromXYZ * from.XYZFromRGB;
        }
    RGBSigmoidPolynomial RGBColorSpace::ToRGBCoeffs(RGB rgb) const {
    DCHECK(rgb.r >= 0 && rgb.g >= 0 && rgb.b >= 0);
    return (*rgbToSpectrumTable)(ClampZero(rgb));
}

const RGBColorSpace *RGBColorSpace::GetNamed(std::string n) {
    std::string name;
    std::transform(n.begin(), n.end(), std::back_inserter(name), ::tolower);
    if (name == "aces2065-1")
        return ACES2065_1;
    else if (name == "rec2020")
        return Rec2020;
    else if (name == "dci-p3")
        return DCI_P3;
    else if (name == "srgb")
        return sRGB;
    else
        return nullptr;
}

const RGBColorSpace *RGBColorSpace::Lookup(Point2f r, Point2f g, Point2f b, Point2f w) {
    auto closeEnough = [](const Point2f &a, const Point2f &b) {
        return ((a.x == b.x || std::abs((a.x - b.x) / b.x) < 1e-3) &&
                (a.y == b.y || std::abs((a.y - b.y) / b.y) < 1e-3));
    };
    for (const RGBColorSpace *cs : {ACES2065_1, DCI_P3, Rec2020, sRGB}) {
        if (closeEnough(r, cs->r) && closeEnough(g, cs->g) && closeEnough(b, cs->b) &&
            closeEnough(w, cs->w))
            return cs;
    }
    return nullptr;
}

const RGBColorSpace *RGBColorSpace::sRGB;
const RGBColorSpace *RGBColorSpace::DCI_P3;
const RGBColorSpace *RGBColorSpace::Rec2020;
const RGBColorSpace *RGBColorSpace::ACES2065_1;


void RGBColorSpace::Init(Allocator alloc) {
    // Rec. ITU-R BT.709.3
    sRGB = alloc.new_object<RGBColorSpace>(
        Point2f(.64, .33), Point2f(.3, .6), Point2f(.15, .06),
        GetNamedSpectrum("stdillum-D65"), RGBToSpectrumTable::sRGB, alloc);
    // P3-D65 (display)
    DCI_P3 = alloc.new_object<RGBColorSpace>(
        Point2f(.68, .32), Point2f(.265, .690), Point2f(.15, .06),
        GetNamedSpectrum("stdillum-D65"), RGBToSpectrumTable::DCI_P3, alloc);
    // ITU-R Rec BT.2020
    Rec2020 = alloc.new_object<RGBColorSpace>(
        Point2f(.708, .292), Point2f(.170, .797), Point2f(.131, .046),
        GetNamedSpectrum("stdillum-D65"), RGBToSpectrumTable::Rec2020, alloc);
    ACES2065_1 = alloc.new_object<RGBColorSpace>(
        Point2f(.7347, .2653), Point2f(0., 1.), Point2f(.0001, -.077),
        GetNamedSpectrum("illum-acesD60"), RGBToSpectrumTable::ACES2065_1, alloc);
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        CUDA_CHECK(cudaMemcpyToSymbol(RGBColorSpace_sRGB, &RGBColorSpace::sRGB,
                                      sizeof(RGBColorSpace_sRGB)));
        CUDA_CHECK(cudaMemcpyToSymbol(RGBColorSpace_DCI_P3, &RGBColorSpace::DCI_P3,
                                      sizeof(RGBColorSpace_DCI_P3)));
        CUDA_CHECK(cudaMemcpyToSymbol(RGBColorSpace_Rec2020, &RGBColorSpace::Rec2020,
                                      sizeof(RGBColorSpace_Rec2020)));
        CUDA_CHECK(cudaMemcpyToSymbol(RGBColorSpace_ACES2065_1,
                                      &RGBColorSpace::ACES2065_1,
                                      sizeof(RGBColorSpace_ACES2065_1)));
    }
#endif
}

// std::string RGBColorSpace::ToString() const {
//     return std::format("[ RGBColorSpace r: {} g: {} b: {} w:{} illuminant: "\
//                         "{} RGBToXYZ: {} XYZToRGB: {} ]",
//                         r.ToString(), g.ToString(), b.ToString(), w.ToString(), illuminant.ToString(), XYZFromRGB.ToString(), RGBFromXYZ.ToString());
// }


}