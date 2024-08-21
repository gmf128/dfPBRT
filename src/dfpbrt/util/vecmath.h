#ifndef DFPBRT_UTIL_VECMATH_H
#define DFPBRT_UTIL_VECMATH_H

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/float.h>
#include <dfpbrt/util/check.h>
#include <dfpbrt/util/log.h>
#include <dfpbrt/util/math.h>

#include <cmath>
#include <array>
#include <span>

namespace dfpbrt {
   namespace{
   //alias
   template<typename T>
   struct TupleLength
   {
        using type = float;
   };
   template <>
   struct TupleLength<double> {
       using type = double;
};

   template <>
   struct TupleLength<long double> {
       using type = long double;
};

   template <>
   struct TupleLength<int> {
       using type = int;
};

   }

   //concepts
   template<typename T, typename U>
   concept Addi_compatibility = std::is_convertible_v<T, decltype(T{}+U{})>;
   template<typename T, typename U>
   concept Multi_compatibility = std::is_convertible_v<T, decltype(T{}*U{})>;

   namespace internal {

          template <typename T>
          std::string ToString2(T x, T y);
          template <typename T>
          std::string ToString3(T x, T y, T z);

}  // namespace internal

     extern template std::string internal::ToString2(float, float);
     extern template std::string internal::ToString2(double, double);
     extern template std::string internal::ToString2(int, int);
     extern template std::string internal::ToString3(float, float, float);
     extern template std::string internal::ToString3(double, double, double);
     extern template std::string internal::ToString3(int, int, int);


   //tuple2
   template<template <typename> typename Child, typename T>
   class Tuple2{
    public:
          static const int nDimensions = 2;
          // Constructor
          Tuple2() = default;
          Tuple2(T x, T y): x(x), y(y){CHECK(HasNaN()); }

          bool HasNaN() const{
               return (IsNaN(x) || IsNaN(y));
          }
          Tuple2(Child<T> c){
               CHECK(!c.HasNaN());
               this.x = c.x;
               this.y = c.y;
          }
          Child<T> &operator=(Child<T> c) {
               DCHECK(!c.HasNaN());
               x = c.x;
               y = c.y;
               return static_cast<Child<T> &>(*this);
          }
          //numerical operations
          template <typename U>
          auto operator+(Child<U> d) const -> Child<decltype(T{}+U{})>{
               CHECK(!d.HasNaN());
               return {x + d.x, y + d.y};
          }

          template <typename U>
          requires Addi_compatibility<T, U>
          Child<T> & operator+=(Child<U> d) {
               CHECK(!d.HasNaN());
               x += d.x;
               y += d.y;
               return static_cast<Child<T> &>(*this);
          }
          template <typename U>
          auto operator-(Child<U> d) const -> Child<decltype(T{}-U{})>{
               CHECK(!d.HasNaN());
               return {x - d.x, y - d.y};
          }

          template <typename U>
          requires Addi_compatibility<T, U>
          Child<T> & operator-=(Child<U> d) {
               CHECK(!d.HasNaN());
               x -= d.x;
               y -= d.y;
               return static_cast<Child<T> &>(*this);
          }

          bool operator==(Child<T> c) const {return (x == c.x) && (y == c.y);}
          bool operator!=(Child<T> c) const {return (x != c.x) || (y != c.y);}

          template<typename U>
          auto operator*(U u) const -> Child<decltype(T{}*U{})>{
               return {x * u, y * u};
          }
          template<typename U>
          requires Multi_compatibility<T, U>
          Child<T> & operator*=(U u) {
               DCHECK(!IsNaN(u));
               x *= u;
               y *= u;
               return static_cast<Child<T> &>(*this);
          }

          template<typename U>
          auto operator/(U d) const -> Child<decltype(T{}/U{})>{
               DCHECK(d!=0 && !IsNaN(d));
               return {x / d, y / d};
          }
          template<typename U>
          Child<T> & operator/=(U d) {
               DCHECK(d!=0 && !IsNaN(d));
               x /= d;
               y /= d;
               return static_cast<Child<T> &>(*this);
          }

           
          Child<T> operator-() const { return {-x, -y}; }

    
          T operator[](int i) const {
               CHECK(i >= 0 && i <= 1);
               return (i == 0) ? x : y;
          }

          T &operator[](int i) {
               CHECK(i >= 0 && i <= 1);
               return (i == 0) ? x : y;
          }

          std::string ToString() const{
               return internal::ToString2(x, y);
          }


        // data member: member-declarator-list 
        T x{}, y{};
   };

     // Tuple2 Inline Functions
template <template <class> class C, typename T, typename U>
inline auto operator*(U s, Tuple2<C, T> t) -> C<decltype(T{} * U{})> {
    CHECK(!t.HasNaN());
    return t * s;
}

template <template <class> class C, typename T>
inline C<T> Abs(Tuple2<C, T> t) {
    // "argument-dependent lookup..." (here and elsewhere)
    using std::abs;
    return {abs(t.x), abs(t.y)};
}

template <template <class> class C, typename T>
inline C<T> Ceil(Tuple2<C, T> t) {
    using std::ceil;
    return {ceil(t.x), ceil(t.y)};
}

template <template <class> class C, typename T>
inline C<T> Floor(Tuple2<C, T> t) {
    using std::floor;
    return {floor(t.x), floor(t.y)};
}

template <template <class> class C, typename T>
inline auto Lerp(Float t, Tuple2<C, T> t0, Tuple2<C, T> t1) {
    return (1 - t) * t0 + t * t1;
}

template <template <class> class C, typename T>
inline C<T> FMA(Float a, Tuple2<C, T> b, Tuple2<C, T> c) {
    return {std::fma(a, b.x, c.x), std::fma(a, b.y, c.y)};
}

template <template <class> class C, typename T>
inline C<T> FMA(Tuple2<C, T> a, Float b, Tuple2<C, T> c) {
    return FMA(b, a, c);
}

template <template <class> class C, typename T>
inline C<T> Min(Tuple2<C, T> t0, Tuple2<C, T> t1) {
    using std::min;
    return {min(t0.x, t1.x), min(t0.y, t1.y)};
}

template <template <class> class C, typename T>
inline T MinComponentValue(Tuple2<C, T> t) {
    using std::min;
    return min(t.x, t.y);
}

template <template <class> class C, typename T>
inline int MinComponentIndex(Tuple2<C, T> t) {
    return (t.x < t.y) ? 0 : 1;
}

template <template <class> class C, typename T>
inline C<T> Max(Tuple2<C, T> t0, Tuple2<C, T> t1) {
    using std::max;
    return {max(t0.x, t1.x), max(t0.y, t1.y)};
}

template <template <class> class C, typename T>
inline T MaxComponentValue(Tuple2<C, T> t) {
    using std::max;
    return max(t.x, t.y);
}

template <template <class> class C, typename T>
inline int MaxComponentIndex(Tuple2<C, T> t) {
    return (t.x > t.y) ? 0 : 1;
}

template <template <class> class C, typename T>
inline C<T> Permute(Tuple2<C, T> t, std::array<int, 2> p) {
    return {t[p[0]], t[p[1]]};
}

template <template <class> class C, typename T>
inline T HProd(Tuple2<C, T> t) {
    return t.x * t.y;
}

//Tuple3, similar to Tuple2
template<template <typename> typename Child, typename T>
   class Tuple3{
    public:
          static const int nDimensions = 3;
          // Constructor
          Tuple3() = default;
          Tuple3(T x, T y, T z): x(x), y(y), z(z){CHECK(HasNaN()); }

          bool HasNaN() const{
               return (IsNaN(x) || IsNaN(y) || IsNaN(z));
          }
          Tuple3(Child<T> c){
               CHECK(!c.HasNaN());
               this.x = c.x;
               this.y = c.y;
               this.z = c.z;
          }
          Child<T> &operator=(Child<T> c) {
               DCHECK(!c.HasNaN());
               x = c.x;
               y = c.y;
               z = c.z;
               return static_cast<Child<T> &>(*this);
          }
          //numerical operations
          template <typename U>
          auto operator+(Child<U> d) const -> Child<decltype(T{}+U{})>{
               CHECK(!d.HasNaN());
               return {x + d.x, y + d.y, z + d.z};
          }

          template <typename U>
          requires Addi_compatibility<T, U>
          Child<T> & operator+=(Child<U> d) {
               CHECK(!d.HasNaN());
               x += d.x;
               y += d.y;
               z += d.z;
               return static_cast<Child<T> &>(*this);
          }
          template <typename U>
          auto operator-(Child<U> d) const -> Child<decltype(T{}-U{})>{
               CHECK(!d.HasNaN());
               return {x - d.x, y - d.y, z-d.z};
          }

          template <typename U>
          requires Addi_compatibility<T, U>
          Child<T> & operator-=(Child<U> d) {
               CHECK(!d.HasNaN());
               x -= d.x;
               y -= d.y;
               z -= d.z;
               return static_cast<Child<T> &>(*this);
          }

          bool operator==(Child<T> c) const {return (x == c.x) && (y == c.y) && (z == c.z);}
          bool operator!=(Child<T> c) const {return (x != c.x) || (y != c.y) || (z != c.z);}

          template<typename U>
          requires Multi_compatibility<T, U>
          auto operator*(U d) const -> Child<decltype(T{}*U{})>{
               CHECK(!IsNaN(d));
               return {x * d, y * d, z * d};
          }
          template<typename U>
          Child<T> & operator*=(U d) {
               CHECK(!IsNaN(d));
               x *= d;
               y *= d;
               z *= d;
               return static_cast<Child<T> &>(*this);
          }

          template<typename U>
          auto operator/(U d) const -> Child<decltype(T{}/U{})>{
               CHECK(d!=0 && !IsNaN(d));
               return {x / d, y / d, z / d};
          }
          template<typename U>
          Child<T> & operator/=(U d) {
               CHECK(d!=0 && !IsNaN(d));
               x /= d;
               y /= d;
               z /= d;
               return static_cast<Child<T> &>(*this);
          }

           
          Child<T> operator-() const { return {-x, -y, -z}; }

    
          T operator[](int i) const {
               CHECK(i >= 0 && i <= 2);
               return (i == 0) ? x : ( i==1 ? y : z);
          }

          T &operator[](int i) {
               CHECK(i >= 0 && i <= 2);
               return (i == 0) ? x : ( i==1 ? y : z);
          }

          std::string ToString() const{
               return internal::ToString3(x, y, z);
          }


        // data member: member-declarator-list 
        T x{}, y{}, z{};
   };

     // Tuple2 Inline Functions
template <template <class> class C, typename T, typename U>
inline auto operator*(U s, Tuple3<C, T> t) -> C<decltype(T{} * U{})> {
    CHECK(!t.HasNaN());
    return t * s;
}

template <template <class> class C, typename T>
inline C<T> Abs(Tuple3<C, T> t) {
    // "argument-dependent lookup..." (here and elsewhere)
    using std::abs;
    return {abs(t.x), abs(t.y), abs(t.z)};
}

template <template <class> class C, typename T>
inline C<T> Ceil(Tuple3<C, T> t) {
    using std::ceil;
    return {ceil(t.x), ceil(t.y), ceil(t.z)};
}

template <template <class> class C, typename T>
inline C<T> Floor(Tuple3<C, T> t) {
    using std::floor;
    return {floor(t.x), floor(t.y), floor(t.z)};
}

template <template <class> class C, typename T>
inline auto Lerp(Float t, Tuple3<C, T> t0, Tuple3<C, T> t1) {
    return (1 - t) * t0 + t * t1;
}

template <template <class> class C, typename T>
inline C<T> FMA(Float a, Tuple3<C, T> b, Tuple3<C, T> c) {
    return {std::fma(a, b.x, c.x), std::fma(a, b.y, c.y), std::fma(a, b.z, c.z)};
}

template <template <class> class C, typename T>
inline C<T> FMA(Tuple3<C, T> a, Float b, Tuple3<C, T> c) {
    return FMA(b, a, c);
}

template <template <class> class C, typename T>
inline C<T> Min(Tuple3<C, T> t0, Tuple3<C, T> t1) {
    using std::min;
    return {min(t0.x, t1.x), min(t0.y, t1.y), min(t0.z, t1.z)};
}

template <template <class> class C, typename T>
inline T MinComponentValue(Tuple3<C, T> t) {
    using std::min;
    return min(min(t.x, t.y), t.z);
}

template <template <class> class C, typename T>
inline int MinComponentIndex(Tuple3<C, T> t) {
     int component = (t.x < t.y) ? 0 : 1;
     return (t[component] < t.z) ? component : 2;
}

template <template <class> class C, typename T>
inline C<T> Max(Tuple3<C, T> t0, Tuple3<C, T> t1) {
    using std::max;
    return {max(t0.x, t1.x), max(t0.y, t1.y), max(t0.z, t1.z)};
}

template <template <class> class C, typename T>
inline T MaxComponentValue(Tuple3<C, T> t) {
    using std::max;
    return max(max(t.x, t.y), t.z);
}

template <template <class> class C, typename T>
inline int MaxComponentIndex(Tuple3<C, T> t) {
    int component = (t.x > t.y) ? 0 : 1;
    return (t[component] > t.z) ? component : 2;
}

template <template <class> class C, typename T>
inline C<T> Permute(Tuple3<C, T> t, std::array<int, 3> p) {
    return {t[p[0]], t[p[1]], t[p[2]]};
}

template <template <class> class C, typename T>
inline T HProd(Tuple3<C, T> t) {
    return t.x * t.y * t.z;
}


template<typename T>
class Point2;
template<typename T>
class Point3;
template<typename T>
class Normal3;

  
// Vector2 Definition
template <typename T>
class Vector2 : public Tuple2<Vector2, T> {
  public:
    // Vector2 Public Methods
    using Tuple2<Vector2, T>::x;
    using Tuple2<Vector2, T>::y;

    Vector2() = default;
    Vector2(T x, T y) : Tuple2<dfpbrt::Vector2, T>(x, y) {}
    template <typename U>
    explicit Vector2(Point2<U> p);
    template <typename U>
    explicit Vector2(Vector2<U> v)
        : Tuple2<dfpbrt::Vector2, T>(T(v.x), T(v.y)) {}
};

// Vector3 Definition
template <typename T>
class Vector3 : public Tuple3<Vector3, T> {
  public:
    // Vector3 Public Methods
    using Tuple3<Vector3, T>::x;
    using Tuple3<Vector3, T>::y;
    using Tuple3<Vector3, T>::z;

    Vector3() = default;
    
    Vector3(T x, T y, T z) : Tuple3<dfpbrt::Vector3, T>(x, y, z) {}

    template <typename U>
    explicit Vector3(Vector3<U> v)
        : Tuple3<dfpbrt::Vector3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    explicit Vector3(Point3<U> p);
    template <typename U>
    explicit Vector3(Normal3<U> n);
};
// Vector2* Definitions
using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;

// Vector3* Definitions
using Vector3f = Vector3<Float>;
using Vector3i = Vector3<int>;


// Point2 Definition
template <typename T>
class Point2 : public Tuple2<Point2, T> {
  public:
    // Point2 Public Methods
    using Tuple2<Point2, T>::x;
    using Tuple2<Point2, T>::y;
    using Tuple2<Point2, T>::HasNaN;
    //Point + Point = Point ok
    using Tuple2<Point2, T>::operator+;
    using Tuple2<Point2, T>::operator+=;
    //Point * Point = Point ok
    using Tuple2<Point2, T>::operator*;
    using Tuple2<Point2, T>::operator*=;

    Point2() { x = y = 0; }
    
    Point2(T x, T y) : Tuple2<dfpbrt::Point2, T>(x, y) {}
    template <typename U>
    explicit Point2(Point2<U> v) : Tuple2<dfpbrt::Point2, T>(T(v.x), T(v.y)) {}
    template <typename U>
    explicit Point2(Vector2<U> v)
        : Tuple2<dfpbrt::Point2, T>(T(v.x), T(v.y)) {}

    template <typename U>
    auto operator+(Vector2<U> v) const -> Point2<decltype(T{} + U{})> {
        DCHECK(!v.HasNaN());
        return {x + v.x, y + v.y};
    }
    template <typename U>
    requires Addi_compatibility<T, U>
    Point2<T> &operator+=(Vector2<U> v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        return *this;
    }

    // We can't do using operator- above, since we don't want to pull in
    // the Point-Point -> Point one so that we can return a vector
    // instead...

    Point2<T> operator-() const { return {-x, -y}; }

    template <typename U>
    auto operator-(Point2<U> p) const -> Vector2<decltype(T{} - U{})> {
        DCHECK(!p.HasNaN());
        return {x - p.x, y - p.y};
    }
    template <typename U>
    auto operator-(Vector2<U> v) const -> Point2<decltype(T{} - U{})> {
        DCHECK(!v.HasNaN());
        return {x - v.x, y - v.y};
    }
    template <typename U>
    Point2<T> &operator-=(Vector2<U> v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        return *this;
    }

// Point2* Definitions
using Point2f = Point2<Float>;
using Point2i = Point2<int>;



// https://www.iquilezles.org/www/articles/ibilinear/ibilinear.htm,
// with a fix for perfect quads
inline Point2f InvertBilinear(Point2f p, std::span<const Point2f> vert) {
    // The below assumes a quad (vs uv parametric layout) in v....
    Point2f a = vert[0], b = vert[1], c = vert[3], d = vert[2];
    Vector2f e = b - a, f = d - a, g = (a - b) + (c - d), h = p - a;

    auto cross2d = [](Vector2f a, Vector2f b) {
        return DifferenceOfProducts(a.x, b.y, a.y, b.x);
    };

    Float k2 = cross2d(g, f);
    Float k1 = cross2d(e, f) + cross2d(h, g);
    Float k0 = cross2d(h, e);

    // if edges are parallel, this is a linear equation
    if (std::abs(k2) < 0.001f) {
        if (std::abs(e.x * k1 - g.x * k0) < 1e-5f)
            return Point2f((h.y * k1 + f.y * k0) / (e.y * k1 - g.y * k0), -k0 / k1);
        else
            return Point2f((h.x * k1 + f.x * k0) / (e.x * k1 - g.x * k0), -k0 / k1);
    }

    Float v0, v1;
    if (!Quadratic(k2, k1, k0, &v0, &v1))
        return Point2f(0, 0);

    Float u = (h.x - f.x * v0) / (e.x + g.x * v0);
    if (u < 0 || u > 1 || v0 < 0 || v0 > 1)
        return Point2f((h.x - f.x * v1) / (e.x + g.x * v1), v1);
    return Point2f(u, v0);
}

};

// Point3 Definition
template <typename T>
class Point3 : public Tuple3<Point3, T> {
  public:
    // Point3 Public Methods
    using Tuple3<Point3, T>::x;
    using Tuple3<Point3, T>::y;
    using Tuple3<Point3, T>::z;
    using Tuple3<Point3, T>::HasNaN;
    using Tuple3<Point3, T>::operator+;
    using Tuple3<Point3, T>::operator+=;
    using Tuple3<Point3, T>::operator*;
    using Tuple3<Point3, T>::operator*=;

    Point3() = default;
    
    Point3(T x, T y, T z) : Tuple3<Point3, T>(x, y, z) {}

    // We can't do using operator- above, since we don't want to pull in
    // the Point-Point -> Point one so that we can return a vector
    // instead...
    Point3<T> operator-() const { return {-x, -y, -z}; }

    template <typename U>
    explicit Point3(Point3<U> p)
        : Tuple3<Point3, T>(T(p.x), T(p.y), T(p.z)) {}
    template <typename U>
    explicit Point3(Vector3<U> v)
        : Tuple3<Point3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    auto operator+(Vector3<U> v) const -> Point3<decltype(T{} + U{})> {
        DCHECK(!v.HasNaN());
        return {x + v.x, y + v.y, z + v.z};
    }
    template <typename U>
    Point3<T> &operator+=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    template <typename U>
    auto operator-(Vector3<U> v) const -> Point3<decltype(T{} - U{})> {
        DCHECK(!v.HasNaN());
        return {x - v.x, y - v.y, z - v.z};
    }
    template <typename U>
    Point3<T> &operator-=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    template <typename U>
    auto operator-(Point3<U> p) const -> Vector3<decltype(T{} - U{})> {
        DCHECK(!p.HasNaN());
        return {x - p.x, y - p.y, z - p.z};
    }
};

// Point3* Definitions
using Point3f = Point3<Float>;
using Point3i = Point3<int>;

// Normal3 Definition
template <typename T>
class Normal3 : public Tuple3<Normal3, T> {
  public:
    // Normal3 Public Methods
    using Tuple3<Normal3, T>::x;
    using Tuple3<Normal3, T>::y;
    using Tuple3<Normal3, T>::z;
    using Tuple3<Normal3, T>::HasNaN;
    using Tuple3<Normal3, T>::operator+;
    using Tuple3<Normal3, T>::operator*;
    using Tuple3<Normal3, T>::operator*=;

    Normal3() = default;
    
    Normal3(T x, T y, T z) : Tuple3<Normal3, T>(x, y, z) {}
    template <typename U>
    explicit Normal3<T>(Normal3<U> v)
        : Tuple3<Normal3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    explicit Normal3<T>(Vector3<U> v)
        : Tuple3<Normal3, T>(T(v.x), T(v.y), T(v.z)) {}
};

using Normal3f = Normal3<Float>;



//New!: Quaternion
// Quaternion Definition {Vector3f, float}
class Quaternion {
  public:
    // Quaternion Public Methods
    Quaternion() = default;
    Quaternion(Vector3f v, Float w): v(v), w(w){}

    
    Quaternion &operator+=(Quaternion q) {
        v += q.v;
        w += q.w;
        return *this;
    }

    Quaternion operator+(Quaternion q) const { return {v + q.v, w + q.w}; }
    
    Quaternion &operator-=(Quaternion q) {
        v -= q.v;
        w -= q.w;
        return *this;
    }
    
    Quaternion operator-() const { return {-v, -w}; }
    
    Quaternion operator-(Quaternion q) const { return {v - q.v, w - q.w}; }
    
    Quaternion &operator*=(Float f) {
        v *= f;
        w *= f;
        return *this;
    }
    
    Quaternion operator*(Float f) const { return {v * f, w * f}; }
    
    Quaternion &operator/=(Float f) {
        DCHECK(f!=0);
        v /= f;
        w /= f;
        return *this;
    }
    
    Quaternion operator/(Float f) const {
        DCHECK(f!=0);
        return {v / f, w / f};
    }

    std::string ToString() const;

    // Quaternion Public Members
    Vector3f v;
    Float w = 1;


    // Vector2 Inline Functions
template <typename T>
template <typename U>
Vector2<T>::Vector2(Point2<U> p) : Tuple2<pbrt::Vector2, T>(T(p.x), T(p.y)) {}

template <typename T>
inline auto Dot(Vector2<T> v1, Vector2<T> v2) ->
    typename TupleLength<T>::type {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return SumOfProducts(v1.x, v2.x, v1.y, v2.y);
}

template <typename T>
inline auto AbsDot(Vector2<T> v1, Vector2<T> v2) ->
    typename TupleLength<T>::type {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return std::abs(Dot(v1, v2));
}

template <typename T>
inline auto LengthSquared(Vector2<T> v) -> typename TupleLength<T>::type {
    return Sqr(v.x) + Sqr(v.y);
}

template <typename T>
inline auto Length(Vector2<T> v) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(v));
}

template <typename T>
inline auto Normalize(Vector2<T> v) {
    return v / Length(v);
}

template <typename T>
inline auto Distance(Point2<T> p1, Point2<T> p2) ->
    typename TupleLength<T>::type {
    return Length(p1 - p2);
}

template <typename T>
inline auto DistanceSquared(Point2<T> p1, Point2<T> p2) ->
    typename TupleLength<T>::type {
    return LengthSquared(p1 - p2);
}

// Vector3 Inline Functions
template <typename T>
template <typename U>
Vector3<T>::Vector3(Point3<U> p) : Tuple3<pbrt::Vector3, T>(T(p.x), T(p.y), T(p.z)) {}

template <typename T>
inline Vector3<T> Cross(Vector3<T> v1, Normal3<T> v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return {DifferenceOfProducts(v1.y, v2.z, v1.z, v2.y),
            DifferenceOfProducts(v1.z, v2.x, v1.x, v2.z),
            DifferenceOfProducts(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
inline Vector3<T> Cross(Normal3<T> v1, Vector3<T> v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return {DifferenceOfProducts(v1.y, v2.z, v1.z, v2.y),
            DifferenceOfProducts(v1.z, v2.x, v1.x, v2.z),
            DifferenceOfProducts(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
inline T LengthSquared(Vector3<T> v) {
    return Sqr(v.x) + Sqr(v.y) + Sqr(v.z);
}

template <typename T>
inline auto Length(Vector3<T> v) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(v));
}

template <typename T>
inline auto Normalize(Vector3<T> v) {
    return v / Length(v);
}

template <typename T>
inline T Dot(Vector3<T> v, Vector3<T> w) {
    DCHECK(!v.HasNaN() && !w.HasNaN());
    return v.x * w.x + v.y * w.y + v.z * w.z;
}

// Equivalent to std::acos(Dot(a, b)), but more numerically stable.
// via http://www.plunk.org/~hatch/rightway.html
template <typename T>
inline Float AngleBetween(Vector3<T> v1, Vector3<T> v2) {
    if (Dot(v1, v2) < 0)
        return Pi - 2 * SafeASin(Length(v1 + v2) / 2);
    else
        return 2 * SafeASin(Length(v2 - v1) / 2);
}

template <typename T>
inline T AbsDot(Vector3<T> v1, Vector3<T> v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return std::abs(Dot(v1, v2));
}

template <typename T>
inline Float AngleBetween(Normal3<T> a, Normal3<T> b) {
    if (Dot(a, b) < 0)
        return Pi - 2 * SafeASin(Length(a + b) / 2);
    else
        return 2 * SafeASin(Length(b - a) / 2);
}

//Given v, w; we want to do schimit-orthognize to get v_perp and  w
template <typename T>
inline Vector3<T> GramSchmidt(Vector3<T> v, Vector3<T> w) {
    return v - Dot(v, w) * w;
}

//Cross product
template <typename T>
inline Vector3<T> Cross(Vector3<T> v, Vector3<T> w) {
    DCHECK(!v.HasNaN() && !w.HasNaN());
    return {DifferenceOfProducts(v.y, w.z, v.z, w.y),
            DifferenceOfProducts(v.z, w.x, v.x, w.z),
            DifferenceOfProducts(v.x, w.y, v.y, w.x)};
}

//Important, given a 3D vector v(need to be normalized), we usually want to get a *local* coornidate system using v and 2 more vectors which perp to v
//for details, see: https://pbr-book.org/4ed/Geometry_and_Transformations/Vectors
template <typename T>
inline void CoordinateSystem(Vector3<T> v1, Vector3<T> *v2, Vector3<T> *v3) {
    Float sign = std::copysign(Float(1), v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * Sqr(v1.x) * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + Sqr(v1.y) * a, -v1.y);
}

template <typename T>
PBRT_CPU_GPU inline void CoordinateSystem(Normal3<T> v1, Vector3<T> *v2, Vector3<T> *v3) {
    Float sign = pstd::copysign(Float(1), v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * Sqr(v1.x) * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + Sqr(v1.y) * a, -v1.y);
}

template <typename T>
template <typename U>
Vector3<T>::Vector3(Normal3<U> n) : Tuple3<pbrt::Vector3, T>(T(n.x), T(n.y), T(n.z)) {}

// Point3 Inline Functions
template <typename T>
PBRT_CPU_GPU inline auto Distance(Point3<T> p1, Point3<T> p2) {
    return Length(p1 - p2);
}

template <typename T>
PBRT_CPU_GPU inline auto DistanceSquared(Point3<T> p1, Point3<T> p2) {
    return LengthSquared(p1 - p2);
}

// Normal3 Inline Functions
template <typename T>
PBRT_CPU_GPU inline auto LengthSquared(Normal3<T> n) -> typename TupleLength<T>::type {
    return Sqr(n.x) + Sqr(n.y) + Sqr(n.z);
}

template <typename T>
PBRT_CPU_GPU inline auto Length(Normal3<T> n) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(n));
}

template <typename T>
PBRT_CPU_GPU inline auto Normalize(Normal3<T> n) {
    return n / Length(n);
}

template <typename T>
PBRT_CPU_GPU inline auto Dot(Normal3<T> n, Vector3<T> v) ->
    typename TupleLength<T>::type {
    DCHECK(!n.HasNaN() && !v.HasNaN());
    return FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z));
}

template <typename T>
PBRT_CPU_GPU inline auto Dot(Vector3<T> v, Normal3<T> n) ->
    typename TupleLength<T>::type {
    DCHECK(!v.HasNaN() && !n.HasNaN());
    return FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z));
}

template <typename T>
PBRT_CPU_GPU inline auto Dot(Normal3<T> n1, Normal3<T> n2) ->
    typename TupleLength<T>::type {
    DCHECK(!n1.HasNaN() && !n2.HasNaN());
    return FMA(n1.x, n2.x, SumOfProducts(n1.y, n2.y, n1.z, n2.z));
}

template <typename T>
PBRT_CPU_GPU inline auto AbsDot(Normal3<T> n, Vector3<T> v) ->
    typename TupleLength<T>::type {
    DCHECK(!n.HasNaN() && !v.HasNaN());
    return std::abs(Dot(n, v));
}

template <typename T>
PBRT_CPU_GPU inline auto AbsDot(Vector3<T> v, Normal3<T> n) ->
    typename TupleLength<T>::type {
    using std::abs;
    DCHECK(!v.HasNaN() && !n.HasNaN());
    return abs(Dot(v, n));
}

template <typename T>
PBRT_CPU_GPU inline auto AbsDot(Normal3<T> n1, Normal3<T> n2) ->
    typename TupleLength<T>::type {
    using std::abs;
    DCHECK(!n1.HasNaN() && !n2.HasNaN());
    return abs(Dot(n1, n2));
}

template <typename T>
PBRT_CPU_GPU inline Normal3<T> FaceForward(Normal3<T> n, Vector3<T> v) {
    return (Dot(n, v) < 0.f) ? -n : n;
}

template <typename T>
PBRT_CPU_GPU inline Normal3<T> FaceForward(Normal3<T> n, Normal3<T> n2) {
    return (Dot(n, n2) < 0.f) ? -n : n;
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> FaceForward(Vector3<T> v, Vector3<T> v2) {
    return (Dot(v, v2) < 0.f) ? -v : v;
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> FaceForward(Vector3<T> v, Normal3<T> n2) {
    return (Dot(v, n2) < 0.f) ? -v : v;
}

// Quaternion Inline Functions
PBRT_CPU_GPU
inline Quaternion operator*(Float f, Quaternion q) {
    return q * f;
}

PBRT_CPU_GPU inline Float Dot(Quaternion q1, Quaternion q2) {
    return Dot(q1.v, q2.v) + q1.w * q2.w;
}

PBRT_CPU_GPU inline Float Length(Quaternion q) {
    return std::sqrt(Dot(q, q));
}
PBRT_CPU_GPU inline Quaternion Normalize(Quaternion q) {
    DCHECK_GT(Length(q), 0);
    return q / Length(q);
}

PBRT_CPU_GPU inline Float AngleBetween(Quaternion q1, Quaternion q2) {
    if (Dot(q1, q2) < 0)
        return Pi - 2 * SafeASin(Length(q1 + q2) / 2);
    else
        return 2 * SafeASin(Length(q2 - q1) / 2);
}

// http://www.plunk.org/~hatch/rightway.html
PBRT_CPU_GPU inline Quaternion Slerp(Float t, Quaternion q1, Quaternion q2) {
    Float theta = AngleBetween(q1, q2);
    Float sinThetaOverTheta = SinXOverX(theta);
    return q1 * (1 - t) * SinXOverX((1 - t) * theta) / sinThetaOverTheta +
           q2 * t * SinXOverX(t * theta) / sinThetaOverTheta;
}

};

} // namespace dfpbrt

#endif // DFPBRT_VEC_MATH_H

