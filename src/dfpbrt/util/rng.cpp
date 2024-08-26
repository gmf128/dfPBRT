#include <dfpbrt/util/rng.h>

#include <cinttypes>

namespace dfpbrt {

std::string RNG::ToString() const {
    return std::format("[ RNG state: {}" PRIu64 " inc: {}" PRIu64 " ]", state, inc);
}

}  // n