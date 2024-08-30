#include <dfpbrt/dfpbrt.h>

#include <dfpbrt/cameras.h>
#include <dfpbrt/options.h>

#include <dfpbrt/util/log.h>

namespace dfpbrt{
// CameraTransform Method Definitions
CameraTransform::CameraTransform(const AnimatedTransform &worldFromCamera) {
    switch (Options->renderingSpace) {
    case RenderingCoordinateSystem::Camera: {
        // Compute _worldFromRender_ for camera-space rendering
        Float tMid = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        worldFromRender = worldFromCamera.Interpolate(tMid);
        break;
    }
    case RenderingCoordinateSystem::CameraWorld: {
        // Compute _worldFromRender_ for camera-world space rendering
        Float tMid = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        Point3f pCamera = worldFromCamera(Point3f(0, 0, 0), tMid);
        worldFromRender = Translate(Vector3f(pCamera));
        break;
    }
    case RenderingCoordinateSystem::World: {
        // Compute _worldFromRender_ for world-space rendering
        worldFromRender = Transform();
        break;
    }
    default:
        LOG_FATAL("Unhandled rendering coordinate space");
    }
    LOG_VERBOSE(std::format("World-space position: {}", worldFromRender(Point3f(0, 0, 0)).ToString()));
    // Compute _renderFromCamera_ transformation
    Transform renderFromWorld = Inverse(worldFromRender);
    Transform rfc[2] = {renderFromWorld * worldFromCamera.startTransform,
                        renderFromWorld * worldFromCamera.endTransform};
    renderFromCamera = AnimatedTransform(rfc[0], worldFromCamera.startTime, rfc[1],
                                         worldFromCamera.endTime);
}

std::string CameraTransform::ToString() const {
    return std::format("[ CameraTransform renderFromCamera: {} worldFromRender: {} ]",
                        renderFromCamera.ToString(), worldFromRender.ToString());
}
}