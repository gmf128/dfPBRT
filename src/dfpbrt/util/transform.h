#ifndef DFPBRT_UTIL_TRANSFORM_H
#define DFPBRT_UTIL_TRANSFORM_H


#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/check.h>
#include <dfpbrt/util/vecmath.h>
#include <dfpbrt/util/float.h>
#include <dfpbrt/util/math.h>
#include <dfpbrt/ray.h>

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

    explicit Transform(const Frame &frame);

    const SquareMatrix<4> &GetMatrix() const { return m; }
    const SquareMatrix<4> &GetInverseMatrix() const { return mInv; }

    bool operator==(const Transform &t) const { return t.m == m; }
    bool operator!=(const Transform &t) const { return t.m != m; }
    bool IsIdentity() const { return m.IsIdentity(); }

    //Apply transformations
    template<typename T>
    Vector3<T> operator()(const Vector3<T> &v) const;
    template<typename T>
    Point3<T> operator()(const Point3<T> &p) const;
    template<typename T>
    Normal3<T> operator()(const Normal3<T> &n) const;

    Vector3fi operator()(const Vector3fi &v) const;
    Point3fi operator()(const Point3fi &p) const;

    Ray operator()(const Ray &r,  Float *tmax) const;
    RayDifferential operator()(const RayDifferential &r,  Float *tmax) const;
    Bounds3f operator()(const Bounds3f &b) const;

    //ApplyInverse transformations
    //Apply transformations
    template<typename T>
    Vector3<T> ApplyInverse(Vector3<T> v) const;
    template<typename T>
    Point3<T> ApplyInverse(Point3<T> p) const;
    template<typename T>
    Normal3<T> ApplyInverse(Normal3<T> n) const;

    inline Ray ApplyInverse(const Ray &r, Float *tMax = nullptr) const;
    inline RayDifferential ApplyInverse(const RayDifferential &r,
                                        Float *tMax = nullptr) const;


    //Composition of transformations
    Transform operator*(const Transform &t2) const {
    return Transform(m * t2.m, t2.mInv * mInv);
    }
    
    //boolean traits of transformations
    bool SwapsHandedness() const {
        // to check if the transformation cause swapping of the handedness
        SquareMatrix<3> s(m[0][0], m[0][1], m[0][2],
                        m[1][0], m[1][1], m[1][2],
                        m[2][0], m[2][1], m[2][2]);
        return Determinant(s) < 0;
    }
    bool HasScale(Float tolerance = 1e-3f) const {
        Float la2 = LengthSquared((*this)(Vector3f(1, 0, 0)));
        Float lb2 = LengthSquared((*this)(Vector3f(0, 1, 0)));
        Float lc2 = LengthSquared((*this)(Vector3f(0, 0, 1)));
        return (std::abs(la2 - 1) > tolerance || std::abs(lb2 - 1) > tolerance ||
                std::abs(lc2 - 1) > tolerance);
    }

    std::string ToString() const;
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

// Transform Inline Methods
template <typename T>
inline Point3<T> Transform::operator()(const Point3<T> &p) const {
    T xp = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
    T yp = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
    T zp = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
    T wp = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
inline Vector3<T> Transform::operator()(const Vector3<T> &v) const {
    return Vector3<T>(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                      m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                      m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
}

//Important! Normals need to behave different from points and vectors. Transform(n) should be (M^-1)^T
template <typename T>
inline Normal3<T> Transform::operator()(const Normal3<T> &n) const {
    T x = n.x, y = n.y, z = n.z;
    return Normal3<T>(mInv[0][0] * x + mInv[1][0] * y + mInv[2][0] * z,
                      mInv[0][1] * x + mInv[1][1] * y + mInv[2][1] * z,
                      mInv[0][2] * x + mInv[1][2] * y + mInv[2][2] * z);
}

inline Point3fi Transform::operator()(const Point3fi &p) const {
        Float x = Float(p.x), y = Float(p.y), z = Float(p.z);
        // Compute transformed coordinates from point _(x, y, z)_
        Float xp = (m[0][0] * x + m[0][1] * y) + (m[0][2] * z + m[0][3]);
        Float yp = (m[1][0] * x + m[1][1] * y) + (m[1][2] * z + m[1][3]);
        Float zp = (m[2][0] * x + m[2][1] * y) + (m[2][2] * z + m[2][3]);
        Float wp = (m[3][0] * x + m[3][1] * y) + (m[3][2] * z + m[3][3]);

        // Compute absolute error for transformed point, _pError_
        Vector3f pError;
        if (p.IsExact()) {
            // Compute error for transformed exact _p_
            pError.x = gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) +
                                   std::abs(m[0][2] * z) + std::abs(m[0][3]));
            pError.y = gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) +
                                   std::abs(m[1][2] * z) + std::abs(m[1][3]));
            pError.z = gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) +
                                   std::abs(m[2][2] * z) + std::abs(m[2][3]));

        } else {
            // Compute error for transformed approximate _p_
            Vector3f pInError = p.Error();
            pError.x = (gamma(3) + 1) * (std::abs(m[0][0]) * pInError.x +
                                         std::abs(m[0][1]) * pInError.y +
                                         std::abs(m[0][2]) * pInError.z) +
                       gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) +
                                   std::abs(m[0][2] * z) + std::abs(m[0][3]));
            pError.y = (gamma(3) + 1) * (std::abs(m[1][0]) * pInError.x +
                                         std::abs(m[1][1]) * pInError.y +
                                         std::abs(m[1][2]) * pInError.z) +
                       gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) +
                                   std::abs(m[1][2] * z) + std::abs(m[1][3]));
            pError.z = (gamma(3) + 1) * (std::abs(m[2][0]) * pInError.x +
                                         std::abs(m[2][1]) * pInError.y +
                                         std::abs(m[2][2]) * pInError.z) +
                       gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) +
                                   std::abs(m[2][2] * z) + std::abs(m[2][3]));
        }

        if (wp == 1)
            return Point3fi(Point3f(xp, yp, zp), pError);
        else
            return Point3fi(Point3f(xp, yp, zp), pError) / wp;
    }

inline Vector3fi Transform::operator()(const Vector3fi &v) const {
    Float x = Float(v.x), y = Float(v.y), z = Float(v.z);
    Vector3f vOutError;
    if (v.IsExact()) {
        vOutError.x = gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) +
                                  std::abs(m[0][2] * z));
        vOutError.y = gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) +
                                  std::abs(m[1][2] * z));
        vOutError.z = gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) +
                                  std::abs(m[2][2] * z));
    } else {
        Vector3f vInError = v.Error();
        vOutError.x = (gamma(3) + 1) * (std::abs(m[0][0]) * vInError.x +
                                        std::abs(m[0][1]) * vInError.y +
                                        std::abs(m[0][2]) * vInError.z) +
                      gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) +
                                  std::abs(m[0][2] * z));
        vOutError.y = (gamma(3) + 1) * (std::abs(m[1][0]) * vInError.x +
                                        std::abs(m[1][1]) * vInError.y +
                                        std::abs(m[1][2]) * vInError.z) +
                      gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) +
                                  std::abs(m[1][2] * z));
        vOutError.z = (gamma(3) + 1) * (std::abs(m[2][0]) * vInError.x +
                                        std::abs(m[2][1]) * vInError.y +
                                        std::abs(m[2][2]) * vInError.z) +
                      gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) +
                                  std::abs(m[2][2] * z));
    }

    Float xp = m[0][0] * x + m[0][1] * y + m[0][2] * z;
    Float yp = m[1][0] * x + m[1][1] * y + m[1][2] * z;
    Float zp = m[2][0] * x + m[2][1] * y + m[2][2] * z;

    return Vector3fi(Vector3f(xp, yp, zp), vOutError);
}

inline Ray Transform::operator()(const Ray &r, Float *tMax) const {
    Point3fi o = (*this)(Point3fi(r.o));
    Vector3f d = (*this)(r.d);
    // Offset ray origin to edge of error bounds and compute _tMax_
    if (Float lengthSquared = LengthSquared(d); lengthSquared > 0) {
        Float dt = Dot(Abs(d), o.Error()) / lengthSquared;
        o += d * dt;
        if (tMax)
            *tMax -= dt;
    }

    return Ray(Point3f(o), d, r.time, r.medium);
}

inline RayDifferential Transform::operator()(const RayDifferential &r,
                                             Float *tMax) const {
    Ray tr = (*this)(Ray(r), tMax);
    RayDifferential ret(tr.o, tr.d, tr.time, tr.medium);
    ret.hasDifferentials = r.hasDifferentials;
    ret.rxOrigin = (*this)(r.rxOrigin);
    ret.ryOrigin = (*this)(r.ryOrigin);
    ret.rxDirection = (*this)(r.rxDirection);
    ret.ryDirection = (*this)(r.ryDirection);
    return ret;
}
//bbox
inline Bounds3f Transform::operator()(const Bounds3f &b) const{
    //TODO: better impl
    Bounds3f bt;
    for (int i = 0; i < 8; ++i)
        bt = Union(bt, (*this)(b.Corner(i)));
    return bt;

}

template <typename T>
inline Point3<T> Transform::ApplyInverse(Point3<T> p) const {
    T x = p.x, y = p.y, z = p.z;
    T xp = (mInv[0][0] * x + mInv[0][1] * y) + (mInv[0][2] * z + mInv[0][3]);
    T yp = (mInv[1][0] * x + mInv[1][1] * y) + (mInv[1][2] * z + mInv[1][3]);
    T zp = (mInv[2][0] * x + mInv[2][1] * y) + (mInv[2][2] * z + mInv[2][3]);
    T wp = (mInv[3][0] * x + mInv[3][1] * y) + (mInv[3][2] * z + mInv[3][3]);
    CHECK(wp != 0);
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
inline Vector3<T> Transform::ApplyInverse(Vector3<T> v) const {
    T x = v.x, y = v.y, z = v.z;
    return Vector3<T>(mInv[0][0] * x + mInv[0][1] * y + mInv[0][2] * z,
                      mInv[1][0] * x + mInv[1][1] * y + mInv[1][2] * z,
                      mInv[2][0] * x + mInv[2][1] * y + mInv[2][2] * z);
}

template <typename T>
inline Normal3<T> Transform::ApplyInverse(Normal3<T> n) const {
    T x = n.x, y = n.y, z = n.z;
    return Normal3<T>(m[0][0] * x + m[1][0] * y + m[2][0] * z,
                      m[0][1] * x + m[1][1] * y + m[2][1] * z,
                      m[0][2] * x + m[1][2] * y + m[2][2] * z);
}

inline Ray Transform::ApplyInverse(const Ray &r, Float *tMax) const {
    Point3fi o = ApplyInverse(Point3fi(r.o));
    Vector3f d = ApplyInverse(r.d);
    // Offset ray origin to edge of error bounds and compute _tMax_
    Float lengthSquared = LengthSquared(d);
    if (lengthSquared > 0) {
        Vector3f oError(o.x.Width() / 2, o.y.Width() / 2, o.z.Width() / 2);
        Float dt = Dot(Abs(d), oError) / lengthSquared;
        o += d * dt;
        if (tMax)
            *tMax -= dt;
    }
    return Ray(Point3f(o), d, r.time, r.medium);
}

inline RayDifferential Transform::ApplyInverse(const RayDifferential &r,
                                               Float *tMax) const {
    Ray tr = ApplyInverse(Ray(r), tMax);
    RayDifferential ret(tr.o, tr.d, tr.time, tr.medium);
    ret.hasDifferentials = r.hasDifferentials;
    ret.rxOrigin = ApplyInverse(r.rxOrigin);
    ret.ryOrigin = ApplyInverse(r.ryOrigin);
    ret.rxDirection = ApplyInverse(r.rxDirection);
    ret.ryDirection = ApplyInverse(r.ryDirection);
    return ret;
}

// AnimatedTransform
// A transform that variates from time to time, the class have a time range and each time it can record(interpolate from the begin and the end) a transform.
class AnimatedTransform{
    public:
        //Constructors
        AnimatedTransform() = default;
        explicit AnimatedTransform(Transform startTransform, Float startTime, Transform endTransform, Float endTime): 
                    startTransform(startTransform), startTime(startTime), endTransform(endTransform), endTime(endTime){
                    };
        Transform startTransform, endTransform;
        Float startTime, endTime;
        Transform Interpolate(Float time) const;

        //operators
        Point3f operator()(Point3f v, Float time) const{
            return Interpolate(time)(v);
        }
        std::string ToString() const{};
};

}




#endif