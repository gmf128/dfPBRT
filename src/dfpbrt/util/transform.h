#ifndef DFPBRT_UTIL_TRANSFORM_H
#define DFPBRT_UTIL_TRANSFORM_H


#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/check.h>
#include <dfpbrt/util/vecmath.h>
#include <dfpbrt/util/float.h>
#include <dfpbrt/util/math.h>

namespace dfpbrt{

class Transform{
    public:
    //By default, the transform will be set to Identity since the default SquareMatrix is Indentity matrix
    Transform() = default;
    //reference &matrix: avoid copy
    Transform(const SquareMatrix<4> &matrix): m(matrix){
        std::optional<SquareMatrix<4>> inv_m =  Inverse(matrix);
        if (inv_m.has_value())
        {
            mInv = inv_m.value();
        }else{
            // Initialize _mInv_ with not-a-number values
            // Signaling nan & quiet nan see: https://en.cppreference.com/w/cpp/types/numeric_limits/signaling_NaN
            Float NaN = std::numeric_limits<Float>::has_signaling_NaN
                            ? std::numeric_limits<Float>::signaling_NaN()
                            : std::numeric_limits<Float>::quiet_NaN();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    mInv[i][j] = NaN;
        }
    }
    //construction delegation/delegating constructor: from cpp11
    Transform(const Float mat[4][4]) : Transform(SquareMatrix<4>(mat)) {}

    Transform(const SquareMatrix<4> &m, const SquareMatrix<4> &mInv): m(m), mInv(mInv){}

    const SquareMatrix<4> &GetMatrix() const { return m; }
    const SquareMatrix<4> &GetInverseMatrix() const { return mInv; }

    bool operator==(const Transform &t) const { return t.m == m; }
    bool operator!=(const Transform &t) const { return t.m != m; }
    bool IsIdentity() const { return m.IsIdentity(); }

    //Apply transformations
    template<typename T>
    Vector3<T> operator()(Vector3<T> v) const;
    template<typename T>
    Point3<T> operator()(Point3<T> p) const;
    template<typename T>
    Normal3<T> operator()(Normal3<T> n) const;

    bool HasScale(Float tolerance = 1e-3f) const {
        Float la2 = LengthSquared((*this)(Vector3f(1, 0, 0)));
        Float lb2 = LengthSquared((*this)(Vector3f(0, 1, 0)));
        Float lc2 = LengthSquared((*this)(Vector3f(0, 0, 1)));
        return (std::abs(la2 - 1) > tolerance || std::abs(lb2 - 1) > tolerance ||
                std::abs(lc2 - 1) > tolerance);
    }


    private:
        SquareMatrix<4> m, mInv;
};


// Transform Functions(not inline function why?)
Transform Translate(Vector3f delta);
Transform Scale(Float x, Float y, Float z);
//Important: theta is degree not radian! It is wierd to call Rotate(Pi/2), but natural to call Rotate(90)
Transform RotateX(Float theta);
Transform RotateY(Float theta);
Transform RotateZ(Float theta);
//Given a camera position, **the position being looked at from the camera**, and an “up” direction, 
//the look-at transformation describes a transformation from a left-handed viewing coordinate system where the camera is at the origin looking down the  axis, and the  axis is along the up direction.
Transform LookAt(Point3f pos, Point3f look, Vector3f up);

inline Transform Transpose(const Transform &t) {
    return Transform(Transpose(t.GetMatrix()), Transpose(t.GetInverseMatrix()));
}
inline Transform Inverse(const Transform &t){
    return Transform(t.GetInverseMatrix(), t.GetMatrix());
}
// Arbitary rotation
// See :https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations for why
inline Transform Rotate(Float sinTheta, Float cosTheta, Vector3f axis) {
    Vector3f a = Normalize(axis);
    SquareMatrix<4> m;
    // Compute rotation of first basis vector
    m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m[0][3] = 0;

    // Compute rotations of second and third basis vectors
    m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m[1][3] = 0;

    m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m[2][3] = 0;

    //m[3][i] == Identity[3][i], and Identity is default initialized. So, we dont have to coding m[3][i] here.
    return Transform(m, Transpose(m));
}
inline Transform Rotate(Float theta, Vector3f axis) {
    Float sinTheta = std::sin(Radians(theta));
    Float cosTheta = std::cos(Radians(theta));
    return Rotate(sinTheta, cosTheta, axis);
}

//Also refer to :https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations for why
inline Transform RotateFromTo(Vector3f from, Vector3f to) {
    // Compute intermediate vector for vector reflection
    Vector3f refl;
    if (std::abs(from.x) < 0.72f && std::abs(to.x) < 0.72f)
        refl = Vector3f(1, 0, 0);
    else if (std::abs(from.y) < 0.72f && std::abs(to.y) < 0.72f)
        refl = Vector3f(0, 1, 0);
    else
        refl = Vector3f(0, 0, 1);

    // Initialize matrix _r_ for rotation
    Vector3f u = refl - from, v = refl - to;
    SquareMatrix<4> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            // Initialize matrix element _r[i][j]_
            r[i][j] = ((i == j) ? 1 : 0) - 2 / Dot(u, u) * u[i] * u[j] -
                      2 / Dot(v, v) * v[i] * v[j] +
                      4 * Dot(u, v) / (Dot(u, u) * Dot(v, v)) * v[i] * u[j];

    return Transform(r, Transpose(r));
}

}




#endif