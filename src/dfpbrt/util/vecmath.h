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


     //If we dont define this func, the compiler will be confused when compiling the class Vector3fi=Vector3<Inteval>, since *Inteval* cannot pass the requirements defined in float.h
     inline bool IsNaN(Interval fi) {
          return dfpbrt::IsNaN(Float(fi));
     }


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
               x = c.x;
               y = c.y;
               z = c.z;
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


// Vector3fi Definition
class Vector3fi : public Vector3<Interval> {
  public:
    // Vector3fi Public Methods
    using Vector3<Interval>::x;
    using Vector3<Interval>::y;
    using Vector3<Interval>::z;
    using Vector3<Interval>::HasNaN;
    using Vector3<Interval>::operator+;
    using Vector3<Interval>::operator+=;
    using Vector3<Interval>::operator*;
    using Vector3<Interval>::operator*=;

    Vector3fi() = default;
    
    Vector3fi(Float x, Float y, Float z)
        : Vector3<Interval>(Interval(x), Interval(y), Interval(z)) {}
    
    Vector3fi(Interval x, Interval y, Interval z) : Vector3<Interval>(x, y, z) {}
    
    Vector3fi(Vector3f p)
        : Vector3<Interval>(Interval(p.x), Interval(p.y), Interval(p.z)) {}
    template <typename T>
    explicit Vector3fi(Point3<T> p)
        : Vector3<Interval>(Interval(p.x), Interval(p.y), Interval(p.z)) {}

     Vector3fi(Vector3<Interval> pfi) : Vector3<Interval>(pfi) {}

    
    Vector3fi(Vector3f v, Vector3f e)
        : Vector3<Interval>(Interval::FromValueAndError(v.x, e.x),
                            Interval::FromValueAndError(v.y, e.y),
                            Interval::FromValueAndError(v.z, e.z)) {}

    Vector3f Error() const { return {x.Width() / 2, y.Width() / 2, z.Width() / 2}; }
    
    bool IsExact() const { return x.Width() == 0 && y.Width() == 0 && z.Width() == 0; }
};


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


};

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

// Point3fi Definition
class Point3fi : public Point3<Interval> {
  public:
    using Point3<Interval>::x;
    using Point3<Interval>::y;
    using Point3<Interval>::z;
    using Point3<Interval>::HasNaN;
    using Point3<Interval>::operator+;
    using Point3<Interval>::operator*;
    using Point3<Interval>::operator*=;

    Point3fi() = default;
    
    Point3fi(Interval x, Interval y, Interval z) : Point3<Interval>(x, y, z) {}
    
    Point3fi(Float x, Float y, Float z)
        : Point3<Interval>(Interval(x), Interval(y), Interval(z)) {}
    
    Point3fi(const Point3f &p)
        : Point3<Interval>(Interval(p.x), Interval(p.y), Interval(p.z)) {}
    
    Point3fi(Point3<Interval> p) : Point3<Interval>(p) {}
    
    Point3fi(Point3f p, Vector3f e)
        : Point3<Interval>(Interval::FromValueAndError(p.x, e.x),
                           Interval::FromValueAndError(p.y, e.y),
                           Interval::FromValueAndError(p.z, e.z)) {}

    
    Vector3f Error() const { return {x.Width() / 2, y.Width() / 2, z.Width() / 2}; }
    
    bool IsExact() const { return x.Width() == 0 && y.Width() == 0 && z.Width() == 0; }

    // Meh--can't seem to get these from Point3 via using declarations...
    template <typename U>
    Point3fi operator+(Vector3<U> v) const {
        DCHECK(!v.HasNaN());
        return {x + v.x, y + v.y, z + v.z};
    }
    template <typename U>
    Point3fi &operator+=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    Point3fi operator-() const { return {-x, -y, -z}; }

    template <typename U>
    Point3fi operator-(Point3<U> p) const {
        DCHECK(!p.HasNaN());
        return {x - p.x, y - p.y, z - p.z};
    }
    template <typename U>
    Point3fi operator-(Vector3<U> v) const {
        DCHECK(!v.HasNaN());
        return {x - v.x, y - v.y, z - v.z};
    }
    template <typename U>
    Point3fi &operator-=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
};

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

};
    // Vector2 Inline Functions
template <typename T>
template <typename U>
Vector2<T>::Vector2(Point2<U> p) : Tuple2<dfpbrt::Vector2, T>(T(p.x), T(p.y)) {}

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
Vector3<T>::Vector3(Point3<U> p) : Tuple3<dfpbrt::Vector3, T>(T(p.x), T(p.y), T(p.z)) {}

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
};

template <typename T>
inline void CoordinateSystem(Normal3<T> v1, Vector3<T> *v2, Vector3<T> *v3) {
    Float sign = std::copysign(Float(1), v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * Sqr(v1.x) * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + Sqr(v1.y) * a, -v1.y);
}

template <typename T>
template <typename U>
Vector3<T>::Vector3(Normal3<U> n) : Tuple3<dfpbrt::Vector3, T>(T(n.x), T(n.y), T(n.z)) {}

// Point3 Inline Functions
template <typename T>
inline auto Distance(Point3<T> p1, Point3<T> p2) {
    return Length(p1 - p2);
}

template <typename T>
inline auto DistanceSquared(Point3<T> p1, Point3<T> p2) {
    return LengthSquared(p1 - p2);
}

// Normal3 Inline Functions
template <typename T>
inline auto LengthSquared(Normal3<T> n) -> typename TupleLength<T>::type {
    return Sqr(n.x) + Sqr(n.y) + Sqr(n.z);
}

template <typename T>
inline auto Length(Normal3<T> n) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(n));
}

template <typename T>
inline auto Normalize(Normal3<T> n) {
    return n / Length(n);
}

template <typename T>
inline auto Dot(Normal3<T> n, Vector3<T> v) ->
    typename TupleLength<T>::type {
    DCHECK(!n.HasNaN() && !v.HasNaN());
    return FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z));
}

template <typename T>
inline auto Dot(Vector3<T> v, Normal3<T> n) ->
    typename TupleLength<T>::type {
    DCHECK(!v.HasNaN() && !n.HasNaN());
    return FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z));
}

template <typename T>
inline auto Dot(Normal3<T> n1, Normal3<T> n2) ->
    typename TupleLength<T>::type {
    DCHECK(!n1.HasNaN() && !n2.HasNaN());
    return FMA(n1.x, n2.x, SumOfProducts(n1.y, n2.y, n1.z, n2.z));
}

template <typename T>
inline auto AbsDot(Normal3<T> n, Vector3<T> v) ->
    typename TupleLength<T>::type {
    DCHECK(!n.HasNaN() && !v.HasNaN());
    return std::abs(Dot(n, v));
}

template <typename T>
inline auto AbsDot(Vector3<T> v, Normal3<T> n) ->
    typename TupleLength<T>::type {
    using std::abs;
    DCHECK(!v.HasNaN() && !n.HasNaN());
    return abs(Dot(v, n));
}

template <typename T>
inline auto AbsDot(Normal3<T> n1, Normal3<T> n2) ->
    typename TupleLength<T>::type {
    using std::abs;
    DCHECK(!n1.HasNaN() && !n2.HasNaN());
    return abs(Dot(n1, n2));
}

template <typename T>
inline Normal3<T> FaceForward(Normal3<T> n, Vector3<T> v) {
    return (Dot(n, v) < 0.f) ? -n : n;
}

template <typename T>
inline Normal3<T> FaceForward(Normal3<T> n, Normal3<T> n2) {
    return (Dot(n, n2) < 0.f) ? -n : n;
}

template <typename T>
inline Vector3<T> FaceForward(Vector3<T> v, Vector3<T> v2) {
    return (Dot(v, v2) < 0.f) ? -v : v;
}

template <typename T>
inline Vector3<T> FaceForward(Vector3<T> v, Normal3<T> n2) {
    return (Dot(v, n2) < 0.f) ? -v : v;
}

// Quaternion Inline Functions

inline Quaternion operator*(Float f, Quaternion q) {
    return q * f;
}

inline Float Dot(Quaternion q1, Quaternion q2) {
    return Dot(q1.v, q2.v) + q1.w * q2.w;
}

inline Float Length(Quaternion q) {
    return std::sqrt(Dot(q, q));
}
inline Quaternion Normalize(Quaternion q) {
    DCHECK(Length(q) > 0);
    return q / Length(q);
}

inline Float AngleBetween(Quaternion q1, Quaternion q2) {
    if (Dot(q1, q2) < 0)
        return Pi - 2 * SafeASin(Length(q1 + q2) / 2);
    else
        return 2 * SafeASin(Length(q2 - q1) / 2);
}

// http://www.plunk.org/~hatch/rightway.html
inline Quaternion Slerp(Float t, Quaternion q1, Quaternion q2) {
    Float theta = AngleBetween(q1, q2);
    Float sinThetaOverTheta = SinXOverX(theta);
    return q1 * (1 - t) * SinXOverX((1 - t) * theta) / sinThetaOverTheta +
           q2 * t * SinXOverX(t * theta) / sinThetaOverTheta;
}


// Bounding Boxes

// Bounds2 Definition
template <typename T>
class Bounds2 {
  public:
    // Bounds2 Public Methods
    Bounds2() {
        //default initialization: an invalid box
        //pMin(max, max); pMax(min, min) -> pMax.x < pMin.x && pMax.y < pMin.y
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = (std::numeric_limits<T>::max)();
        pMin = Point2<T>(maxNum, maxNum);
        pMax = Point2<T>(minNum, minNum);
    }
    
    explicit Bounds2(Point2<T> p) : pMin(p), pMax(p) {}
    
    Bounds2(Point2<T> p1, Point2<T> p2) : pMin(Min(p1, p2)), pMax(Max(p1, p2)) {}

    template <typename U>
    explicit Bounds2(const Bounds2<U> &b) {
        if (b.IsEmpty())
            // Be careful about overflowing float->int conversions and the
            // like.
            *this = Bounds2<T>();
        else {
            pMin = Point2<T>(b.pMin);
            pMax = Point2<T>(b.pMax);
        }
    }

    //Diagnal vector
    Vector2<T> Diagonal() const { return pMax - pMin; }

    //2D, so return the area of the rectangle
    T Area() const {
        Vector2<T> d = pMax - pMin;
        return d.x * d.y;
    }

    bool IsEmpty() const { return pMin.x >= pMax.x || pMin.y >= pMax.y; }//Degenerate or nondegenerate but with 0 volume

    bool IsDegenerate() const { return pMin.x > pMax.x || pMin.y > pMax.y; }

    int MaxDimension() const {
        Vector2<T> diag = Diagonal();
        if (diag.x > diag.y)
            return 0;
        else
            return 1;
    }
    
    Point2<T> operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    
    Point2<T> &operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    
    bool operator==(const Bounds2<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    
    bool operator!=(const Bounds2<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    
    Point2<T> Corner(int corner) const {
        DCHECK(corner >= 0 && corner < 4);
        return Point2<T>((*this)[(corner & 1)].x, (*this)[(corner & 2) ? 1 : 0].y);
    }
    
    Point2<T> Lerp(Point2f t) const {
        return Point2<T>(dfpbrt::Lerp(t.x, pMin.x, pMax.x),
                         dfpbrt::Lerp(t.y, pMin.y, pMax.y));
    }
    
    Vector2<T> Offset(Point2<T> p) const {
        Vector2<T> o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        return o;
    }
    
    void BoundingSphere(Point2<T> *c, Float *rad) const {
        *c = (pMin + pMax) / 2;
        *rad = Inside(*c, *this) ? Distance(*c, pMax) : 0;
    }

    std::string ToString() const { return StringPrintf("[ %s - %s ]", pMin, pMax); }

    // Bounds2 Public Members
    Point2<T> pMin, pMax;
};

// Bounds3 Definition
template <typename T>
class Bounds3 {
  public:
    // Bounds3 Public Methods
    
    Bounds3() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = (std::numeric_limits<T>::max)();
        pMin = Point3<T>(maxNum, maxNum, maxNum);
        pMax = Point3<T>(minNum, minNum, minNum);
    }

   
    explicit Bounds3(Point3<T> p) : pMin(p), pMax(p) {}

    
    Bounds3(Point3<T> p1, Point3<T> p2) : pMin(Min(p1, p2)), pMax(Max(p1, p2)) {}

    
    Point3<T> operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    
    Point3<T> &operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }

    
    Point3<T> Corner(int corner) const {
     //Elegant code
        DCHECK(corner >= 0 && corner < 8);
        return Point3<T>((*this)[(corner & 1)].x, (*this)[(corner & 2) ? 1 : 0].y,
                         (*this)[(corner & 4) ? 1 : 0].z);
    }

    
    Vector3<T> Diagonal() const { return pMax - pMin; }

    
    T SurfaceArea() const {
        Vector3<T> d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    
    T Volume() const {
        Vector3<T> d = Diagonal();
        return d.x * d.y * d.z;
    }

    
    int MaxDimension() const {
     //Return the dimension on which the extend is longest, useful in building accelerating structures like SD-Tree
        Vector3<T> d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    
    Point3f Lerp(Point3f t) const {
     //Linear intERPolation
        return Point3f(dfpbrt::Lerp(t.x, pMin.x, pMax.x), dfpbrt::Lerp(t.y, pMin.y, pMax.y),
                       dfpbrt::Lerp(t.z, pMin.z, pMax.z));
    }

    
    Vector3f Offset(Point3f p) const {
     //Inverse of Lerp: get the local coordinates
        Vector3f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    
    void BoundingSphere(Point3<T> *center, Float *radius) const {
        *center = (pMin + pMax) / 2;
        *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
    }

    
    bool IsEmpty() const {
        return pMin.x >= pMax.x || pMin.y >= pMax.y || pMin.z >= pMax.z;
    }
   
    bool IsDegenerate() const {
        return pMin.x > pMax.x || pMin.y > pMax.y || pMin.z > pMax.z;
    }

    template <typename U>
    explicit Bounds3(const Bounds3<U> &b) {
        if (b.IsEmpty())
            // Be careful about overflowing float->int conversions and the
            // like.
            *this = Bounds3<T>();
        else {
            pMin = Point3<T>(b.pMin);
            pMax = Point3<T>(b.pMax);
        }
    }
    
    bool operator==(const Bounds3<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    
    bool operator!=(const Bounds3<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    
    bool IntersectP(Point3f o, Vector3f d, Float tMax = Infinity, Float *hitt0 = nullptr,
                    Float *hitt1 = nullptr) const;
    
    bool IntersectP(Point3f o, Vector3f d, Float tMax, Vector3f invDir,
                    const int dirIsNeg[3]) const;

    std::string ToString() const { return StringPrintf("[ %s - %s ]", pMin, pMax); }

    // Bounds3 Public Members
    Point3<T> pMin, pMax;
};

// Bounds[23][fi] Definitions
using Bounds2f = Bounds2<Float>;
using Bounds2i = Bounds2<int>;
using Bounds3f = Bounds3<Float>;
using Bounds3i = Bounds3<int>;

//For integer bounds, there is an iterator class that fulfills the requirements of a C++ forward iterator (i.e., it can only be advanced). 
//The details are slightly tedious and not particularly interesting, so the code is not included in the book. Having this definition makes it possible to write code using range-based for loops to iterate over integer coordinates in a bounding box:
/** Bounds2i b = ...;
    for (Point2i p : b) {
       //  …
}*/
class Bounds2iIterator : public std::forward_iterator_tag {
  public:
    Bounds2iIterator(const Bounds2i &b, const Point2i &pt) : p(pt), bounds(&b) {}
    //++i
    Bounds2iIterator operator++() {
        advance();
        return *this;
    }
    //i++
    Bounds2iIterator operator++(int) {
        Bounds2iIterator old = *this;
        advance();
        return old;
    }
    
    bool operator==(const Bounds2iIterator &bi) const {
        return p == bi.p && bounds == bi.bounds;
    }
    
    bool operator!=(const Bounds2iIterator &bi) const {
        return p != bi.p || bounds != bi.bounds;
    }

    Point2i operator*() const { return p; }

  private:
    void advance() {
        ++p.x;
        if (p.x == bounds->pMax.x) {
            p.x = bounds->pMin.x;
            ++p.y;
        }
    }
    Point2i p;
    const Bounds2i *bounds;
};

// Bounds2 Inline Functions
template <typename T>
inline Bounds2<T> Union(const Bounds2<T> &b1, const Bounds2<T> &b2) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
inline Bounds2<T> Intersect(const Bounds2<T> &b1, const Bounds2<T> &b2) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> b;
    b.pMin = Max(b1.pMin, b2.pMin);
    b.pMax = Min(b1.pMax, b2.pMax);
    return b;
}

template <typename T>
inline bool Overlaps(const Bounds2<T> &ba, const Bounds2<T> &bb) {
    bool x = (ba.pMax.x >= bb.pMin.x) && (ba.pMin.x <= bb.pMax.x);
    bool y = (ba.pMax.y >= bb.pMin.y) && (ba.pMin.y <= bb.pMax.y);
    return (x && y);
}

template <typename T>
inline bool Inside(Point2<T> pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x <= b.pMax.x && pt.y >= b.pMin.y && pt.y <= b.pMax.y);
}

template <typename T>
inline bool Inside(const Bounds2<T> &ba, const Bounds2<T> &bb) {
    return (ba.pMin.x >= bb.pMin.x && ba.pMax.x <= bb.pMax.x && ba.pMin.y >= bb.pMin.y &&
            ba.pMax.y <= bb.pMax.y);
}

template <typename T>
inline bool InsideExclusive(Point2<T> pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x < b.pMax.x && pt.y >= b.pMin.y && pt.y < b.pMax.y);
}

template <typename T, typename U>
inline Bounds2<T> Expand(const Bounds2<T> &b, U delta) {
    Bounds2<T> ret;
    ret.pMin = b.pMin - Vector2<T>(delta, delta);
    ret.pMax = b.pMax + Vector2<T>(delta, delta);
    return ret;
}

// Bounds3 Inline Functions
template <typename T>
inline Bounds3<T> Union(const Bounds3<T> &b, Point3<T> p) {
    Bounds3<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T>
inline Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    Bounds3<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
inline Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    Bounds3<T> b;
    b.pMin = Max(b1.pMin, b2.pMin);
    b.pMax = Min(b1.pMax, b2.pMax);
    return b;
}

template <typename T>
inline bool Overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
    return (x && y && z);
}

template <typename T>
inline bool Inside(Point3<T> p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y && p.y <= b.pMax.y &&
            p.z >= b.pMin.z && p.z <= b.pMax.z);
}

template <typename T>
inline bool InsideExclusive(Point3<T> p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y && p.y < b.pMax.y &&
            p.z >= b.pMin.z && p.z < b.pMax.z);
}

//https://pbr-book.org/4ed/Geometry_and_Transformations/Bounding_Boxes Gives an illustration
template <typename T, typename U>
inline auto DistanceSquared(Point3<T> p, const Bounds3<U> &b) {
    using TDist = decltype(T{} - U{});
    TDist dx = (std::max<TDist>)({0, b.pMin.x - p.x, p.x - b.pMax.x});
    TDist dy = (std::max<TDist>)({0, b.pMin.y - p.y, p.y - b.pMax.y});
    TDist dz = (std::max<TDist>)({0, b.pMin.z - p.z, p.z - b.pMax.z});
    return Sqr(dx) + Sqr(dy) + Sqr(dz);
}

template <typename T, typename U>
inline auto Distance(Point3<T> p, const Bounds3<U> &b) {
    auto dist2 = DistanceSquared(p, b);
    using TDist = typename TupleLength<decltype(dist2)>::type;
    return std::sqrt(TDist(dist2));
}

template <typename T, typename U>
inline Bounds3<T> Expand(const Bounds3<T> &b, U delta) {
    Bounds3<T> ret;
    ret.pMin = b.pMin - Vector3<T>(delta, delta, delta);
    ret.pMax = b.pMax + Vector3<T>(delta, delta, delta);
    return ret;
}

template <typename T>
inline bool Bounds3<T>::IntersectP(Point3f o, Vector3f d, Float tMax,
                                                Float *hitt0, Float *hitt1) const {
    //TODO:
    return true;
}

template <typename T>
inline bool Bounds3<T>::IntersectP(Point3f o, Vector3f d, Float raytMax,
                                                Vector3f invDir,
                                                const int dirIsNeg[3]) const {
    //TODO:
}


inline Bounds2iIterator begin(const Bounds2i &b) {
    return Bounds2iIterator(b, b.pMin);
}


inline Bounds2iIterator end(const Bounds2i &b) {
    // Normally, the ending point is at the minimum x value and one past
    // the last valid y value.
    Point2i pEnd(b.pMin.x, b.pMax.y);//the last valid y value is b.Max.y-1 . See InsideExclusive
    // However, if the bounds are degenerate, override the end point to
    // equal the start point so that any attempt to iterate over the bounds
    // exits out immediately.
    if (b.pMin.x >= b.pMax.x || b.pMin.y >= b.pMax.y)
        pEnd = b.pMin;
    return Bounds2iIterator(b, pEnd);
}

template <typename T>
inline Bounds2<T> Union(const Bounds2<T> &b, Point2<T> p) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}


// Spherical Geometry Functions
// Cartesian coordinates and spherical coordinates
// unit vector(directions)
inline Vector3f SphericalDirection(Float sinTheta, Float cosTheta, Float phi){
     //Notice that the function is given the sine and cosine of theta, rather than eheta itself. This is because the sine and cosine of theta are often already available to the caller. This is not normally the case for phi, however, so phi is passed in as is.
     DCHECK(sinTheta >= -1.0001 && sinTheta <= 1.0001);
     DCHECK(cosTheta >= -1.0001 && cosTheta <= 1.0001);
     return {  
          Clamp(sinTheta, -1, 1) * std::cos(phi),
          Clamp(sinTheta, -1, 1) * std::sin(phi),
          Clamp(cosTheta, -1, 1)
     };
}
// Then we try to get sintheta, costheta and phi from a normalized vector
inline Float CosTheta(Vector3f v){
     DCHECK(v.z >= -1.0001 && v.z <= 1.0001);
     return v.z;
}

inline Float Cos2Theta(Vector3f v){
     DCHECK(v.z >= -1.0001 && v.z <= 1.0001);
     return Sqr(v.z);
}

inline Float Sin2Theta(Vector3f v){
     return (std::max<Float>)(0, 1-Cos2Theta(v));
}

inline Float SinTheta(Vector3f v){
     return std::sqrt(Sin2Theta(v));
}

inline Float TanTheta(Vector3f v){
     return SinTheta(v) / CosTheta(v);
}

inline Float Tan2Theta(Vector3f v){
     return Sin2Theta(v)/Cos2Theta(v);
}

inline Float SphericalTheta(Vector3f v){
     return SafeACos(CosTheta(v));
}

inline Float SpheriacalPhi(Vector3f v){
     //tan:= sin/cos so, atan2(sin, cos) = atan2(v.y, v.x)
     Float tmp = std::atan2(v.y, v.x);
     return (tmp < 0? tmp + Pi * 2 : tmp);// Phi range [0. 2*Pi)
}

inline Float CosPhi(Vector3f v){
     Float sintheta = SinTheta(v);
     return (sintheta == 0) ? 1 : (v.x / sintheta);
}

inline Float SinPhi(Vector3f v){
     Float sintheta = SinTheta(v);
     return (sintheta == 0) ? 0 : (v.y / sintheta);
}

//Spherical area function(triangle and quadrilateral)
inline Float SphericalTriangleArea(Vector3f a, Vector3f b, Vector3f c) {
    return std::abs(
        2 * std::atan2(Dot(a, Cross(b, c)), 1 + Dot(a, b) + Dot(a, c) + Dot(b, c)));
}

inline Float SphericalQuadArea(Vector3f a, Vector3f b, Vector3f c, Vector3f d){
     //TODO:
}

// Octahedral Encoding
// OctahedralVector Definition
class OctahedralVector {
  public:
    // OctahedralVector Public Methods
    OctahedralVector() = default;
    OctahedralVector(Vector3f v) {
        v /= std::abs(v.x) + std::abs(v.y) + std::abs(v.z);
        if (v.z >= 0) {
            x = Encode(v.x);
            y = Encode(v.y);
        } else {
            // Encode octahedral vector with $z < 0$
            x = Encode((1 - std::abs(v.y)) * Sign(v.x));
            y = Encode((1 - std::abs(v.x)) * Sign(v.y));
        }
    }

    explicit operator Vector3f() const {
        Vector3f v;
        v.x = -1 + 2 * (x / 65535.f);
        v.y = -1 + 2 * (y / 65535.f);
        v.z = 1 - (std::abs(v.x) + std::abs(v.y));
        // Reparameterize directions in the $z<0$ portion of the octahedron
        if (v.z < 0) {
            Float xo = v.x;
            v.x = (1 - std::abs(v.y)) * Sign(xo);
            v.y = (1 - std::abs(xo)) * Sign(v.y);
        }

        return Normalize(v);
    }

    std::string ToString() const {
        //return StringPrintf("[ OctahedralVector x: %d y: %d ]", x, y);
        // using std::format instead
        return std::format("[ OctahedralVector x: %d y: %d ]", x, y);
    }

  private:
    // OctahedralVector Private Methods
    static Float Sign(Float v) { return std::copysign(1.f, v); }

    static uint16_t Encode(Float f) {
        return std::round(Clamp((f + 1) / 2, 0, 1) * 65535.f);
    }

    // OctahedralVector Private Members
    uint16_t x, y;
};




} // namespace dfpbrt

#endif // DFPBRT_VEC_MATH_H

