#ifndef DFPBRT_BASE_FILTER_H
#define DFPBRT_BASE_FILTER_H

#include <dfpbrt/dfpbrt.h>

namespace dfpbrt{

// Filter Declarations
struct FilterSample;
class BoxFilter;
class GaussianFilter;
class MitchellFilter;
class LanczosSincFilter;
class TriangleFilter;

// Filter Definition
class Filter : public TaggedPointer<BoxFilter, GaussianFilter, MitchellFilter,
                                    LanczosSincFilter, TriangleFilter> {
  public:
    // Filter Interface
    using TaggedPointer::TaggedPointer;

    static Filter Create(const std::string &name, const ParameterDictionary &parameters,
                         const FileLoc *loc, Allocator alloc);

    DFPBRT_CPU_GPU inline Vector2f Radius() const;

    DFPBRT_CPU_GPU inline Float Evaluate(Point2f p) const;

    DFPBRT_CPU_GPU inline Float Integral() const;

    DFPBRT_CPU_GPU inline FilterSample Sample(Point2f u) const;

    std::string ToString() const;
};
};

#endif