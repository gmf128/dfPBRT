#include <dfpbrt/options.h>

namespace dfpbrt{

    std::string ToString(const RenderingCoordinateSystem & rs){
        switch (rs)
        {
        case RenderingCoordinateSystem::Camera:
            return "Camera";
            break;
        case RenderingCoordinateSystem::CameraWorld:
            return "CameraWorld";
            break;
        default:
            return "World";
            break;
        }
    };
}