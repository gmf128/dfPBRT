#include<dfpbrt/dfpbrt.h>
#include<dfpbrt/util/math.h>

#include<dfpbrt/util/vecmath.h>
#include<dfpbrt/util/log.h>

namespace dfpbrt{
    template class SquareMatrix<2>;
    template class SquareMatrix<3>;
    template class SquareMatrix<4>;

    std::string SquareMatrix<3>::ToString() const{
        return std::format("{} {} {}\n {} {} {}\n{} {} {}\n", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
    }
    // Square--Sphere Mapping Function Definitions
    // Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
    Vector3f EqualAreaSquareToSphere(Point2f p) {
        CHECK(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        // Transform _p_ to $[-1,1]^2$ and compute absolute values
        Float u = 2 * p.x - 1, v = 2 * p.y - 1;
        Float up = std::abs(u), vp = std::abs(v);

        // Compute radius _r_ as signed distance from diagonal
        Float signedDistance = 1 - (up + vp);
        Float d = std::abs(signedDistance);
        Float r = 1 - d;

        // Compute angle $\phi$ for square to sphere mapping
        Float phi = (r == 0 ? 1 : (vp - up) / r + 1) * Pi / 4;

        // Find $z$ coordinate for spherical direction
        Float z = std::copysign(1 - Sqr(r), signedDistance);

        // Compute $\cos\phi$ and $\sin\phi$ for original quadrant and return vector
        Float cosPhi = std::copysign(std::cos(phi), u);
        Float sinPhi = std::copysign(std::sin(phi), v);
        return Vector3f(cosPhi * r * SafeSqrt(2 - Sqr(r)), sinPhi * r * SafeSqrt(2 - Sqr(r)),
                        z);
}


    // Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
    Point2f EqualAreaSphereToSquare(Vector3f d) {
        DCHECK(LengthSquared(d) > .999 && LengthSquared(d) < 1.001);
        Float x = std::abs(d.x), y = std::abs(d.y), z = std::abs(d.z);

        // Compute the radius r
        Float r = SafeSqrt(1 - z);  // r = sqrt(1-|z|)

        // Compute the argument to atan (detect a=0 to avoid div-by-zero)
        Float a = std::max(x, y), b = std::min(x, y);
        b = a == 0 ? 0 : b / a;

        // Polynomial approximation of atan(x)*2/pi, x=b
        // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
        // x=[0,1].
        const Float t1 = 0.406758566246788489601959989e-5;
        const Float t2 = 0.636226545274016134946890922156;
        const Float t3 = 0.61572017898280213493197203466e-2;
        const Float t4 = -0.247333733281268944196501420480;
        const Float t5 = 0.881770664775316294736387951347e-1;
        const Float t6 = 0.419038818029165735901852432784e-1;
        const Float t7 = -0.251390972343483509333252996350e-1;
        Float phi = EvaluatePolynomial(b, t1, t2, t3, t4, t5, t6, t7);

        // Extend phi if the input is in the range 45-90 degrees (u<v)
        if (x < y)
            phi = 1 - phi;

        // Find (u,v) based on (r,phi)
        Float v = phi * r;
        Float u = r - v;

        if (d.z < 0) {
            // southern hemisphere -> mirror u,v
            std::swap(u, v);
            u = 1 - u;
            v = 1 - v;
    }

        // Move (u,v) to the correct quadrant based on the signs of (x,y)
        u = std::copysign(u, d.x);
        v = std::copysign(v, d.y);

        // Transform (u,v) from [-1,1] to [0,1]
        return Point2f(0.5f * (u + 1), 0.5f * (v + 1));
}

    Point2f WrapEqualAreaSquare(Point2f uv) {
    if (uv[0] < 0) {
        uv[0] = -uv[0];     // mirror across u = 0
        uv[1] = 1 - uv[1];  // mirror across v = 0.5
    } else if (uv[0] > 1) {
        uv[0] = 2 - uv[0];  // mirror across u = 1
        uv[1] = 1 - uv[1];  // mirror across v = 0.5
    }
    if (uv[1] < 0) {
        uv[0] = 1 - uv[0];  // mirror across u = 0.5
        uv[1] = -uv[1];     // mirror across v = 0;
    } else if (uv[1] > 1) {
        uv[0] = 1 - uv[0];  // mirror across u = 0.5
        uv[1] = 2 - uv[1];  // mirror across v = 1
    }
    return uv;
}
}