#ifndef DFPBRT_BASE_SAMPLER_H
#define DFPBRT_BASE_SAMPLER_H

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/taggedptr.h>
#include <dfpbrt/util/vecmath.h>

namespace dfpbrt{
struct CameraSample
{
    /* data */
    // pFilm member gives the point on the film to which the generated ray should carry radiance. 
    Point2f pFilm;
    // The point on the lens the ray passes through is in pLens (for cameras that include the notion of lenses),
    Point2f pLens;
    // Time gives the time at which the ray should sample the scene. If the camera itself is in motion, the time value determines what camera position to use when generating the ray.
    Float time = 0;
    // Reconstruciton filterWeight
    Float filterWeight = 1;
    std::string ToString() const;
};




}

#endif