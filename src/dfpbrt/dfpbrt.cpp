#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/util/color.h>
#include <dfpbrt/util/colorspace.h>
#include <dfpbrt/util/spectrum.h>
#include <dfpbrt/options.h>

#include <memory_resource>

namespace dfpbrt{

void InitDFPBRT(const DFPBRTOptions & options){
    Options = new DFPBRTOptions(options);
    // Leak this so memory it allocates isn't freed
    std::pmr::memory_resource* bufferResource = new std::pmr::monotonic_buffer_resource(1024 * 1024);
    Allocator alloc(bufferResource);
    ColorEncoding::Init(alloc);
    Spectra::Init(alloc);
    RGBToSpectrumTable::Init(alloc);

    RGBColorSpace::Init(alloc);
}

}