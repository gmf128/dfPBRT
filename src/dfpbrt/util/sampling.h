#ifndef DFPBRT_UTIL_SAMPLING_H
#define DFPBRT_UTIL_SAMPLING_H

#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/rng.h>

#include <dfpbrt/util/math.h>

namespace dfpbrt{
    namespace detail{

template <typename Iterator>
class IndexingIterator {
  public:
    template <typename Generator>
    DFPBRT_CPU_GPU IndexingIterator(int i, int n, const Generator *) : i(i), n(n) {}

    DFPBRT_CPU_GPU
    bool operator==(const Iterator &it) const { return i == it.i; }
    DFPBRT_CPU_GPU
    bool operator!=(const Iterator &it) const { return !(*this == it); }
    DFPBRT_CPU_GPU
    Iterator &operator++() {
        ++i;
        return (Iterator &)*this;
    }
    DFPBRT_CPU_GPU
    Iterator operator++(int) const {
        Iterator it = *this;
        return ++it;
    }

  protected:
    int i, n;
};

template <typename Generator, typename Iterator>
class IndexingGenerator {
  public:
    DFPBRT_CPU_GPU
    IndexingGenerator(int n) : n(n) {}
    DFPBRT_CPU_GPU
    Iterator begin() const { return Iterator(0, n, (const Generator *)this); }
    DFPBRT_CPU_GPU
    Iterator end() const { return Iterator(n, n, (const Generator *)this); }

  protected:
    int n;
};


class Stratified1DIter;
template <typename Generator, typename Iterator>
class RNGGenerator;

template <typename Iterator>
class RNGIterator : public IndexingIterator<Iterator> {
  public:
    template <typename Generator>
    DFPBRT_CPU_GPU RNGIterator(int i, int n,
                             const RNGGenerator<Generator, Iterator> *generator)
        : IndexingIterator<Iterator>(i, n, generator), rng(generator->sequenceIndex) {}

  protected:
    RNG rng;
};

//RNGGEnerator definition
template <typename Generator, typename Iterator>
class RNGGenerator : public IndexingGenerator<Generator, Iterator> {
  public:
    DFPBRT_CPU_GPU
    RNGGenerator(int n, uint64_t sequenceIndex = 0, uint64_t seed = PCG32_DEFAULT_STATE)
        : IndexingGenerator<Generator, Iterator>(n),
          sequenceIndex(sequenceIndex),
          seed(seed) {}

  protected:
    friend RNGIterator<Iterator>;
    uint64_t sequenceIndex, seed;
};

class Stratified1DIter : public RNGIterator<Stratified1DIter> {
  public:
    using RNGIterator<Stratified1DIter>::RNGIterator;
    DFPBRT_CPU_GPU
    Float operator*() { return (i + rng.Uniform<Float>()) / n; }
};

}
    
class Stratified1D : public detail::RNGGenerator<Stratified1D, detail::Stratified1DIter> {
  public:
    using detail::RNGGenerator<Stratified1D, detail::Stratified1DIter>::RNGGenerator;
};

inline Point2f SampleUniformDiskConcentric(Point2f u) {
    // Map _u_ to $[-1,1]^2$ and handle degeneracy at the origin
    Point2f uOffset = 2 * u - Vector2f(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
        return {0, 0};

    // Apply concentric mapping to point
    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * Point2f(std::cos(theta), std::sin(theta));
}






    Float SampleVisibleWavelengths(Float u);
    Float VisibleWavelengthsPDF(Float value);
}


#endif