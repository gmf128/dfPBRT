#include <dfpbrt/util/transform.h>

namespace dfpbrt{
    Transform Translate(Vector3f delta){
        SquareMatrix<4> m 
        = {
            1, 0, 0, delta.x,
            0, 1, 0, delta.y,
            0, 0, 1, delta.z,
            0, 0, 0, 1
        };
        SquareMatrix<4> mInv = {
            1, 0, 0, -delta.x,
            0, 1, 0, -delta.y,
            0, 0, 1, -delta.z,
            0, 0, 0, 1
        };
        return Transform(m, mInv);
    }
    Transform Scale(Float x, Float y, Float z){
        SquareMatrix<4> m 
        = {
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1
        };
        SquareMatrix<4> mInv = {
            1.f/x, 0, 0, 0,
            0, 1.f/y, 0, 0,
            0, 0, 1.f/z, 0,
            0, 0, 0, 1
        };
        return Transform(m, mInv);
    }
    //Important: theta is degree not radian! It is wierd to call Rotate(Pi/2), but natural to call Rotate(90)
    Transform RotateX(Float theta) {
        Float sinTheta = std::sin(Radians(theta));
        Float cosTheta = std::cos(Radians(theta));
        SquareMatrix<4> m(1,        0,         0, 0,
                        0, cosTheta, -sinTheta, 0,
                        0, sinTheta,  cosTheta, 0,
                        0,        0,         0, 1);
        return Transform(m, Transpose(m));
    }

    Transform RotateY(Float theta) {
        Float sinTheta = std::sin(Radians(theta));
        Float cosTheta = std::cos(Radians(theta));
        SquareMatrix<4> m( cosTheta, 0, sinTheta, 0,
                                0, 1,        0, 0,
                        -sinTheta, 0, cosTheta, 0,
                                0, 0,        0, 1);
        return Transform(m, Transpose(m));
        }


    Transform RotateZ(Float theta) {
        Float sinTheta = std::sin(Radians(theta));
        Float cosTheta = std::cos(Radians(theta));
        SquareMatrix<4> m(cosTheta, -sinTheta, 0, 0,
                      sinTheta,  cosTheta, 0, 0,
                             0,         0, 1, 0,
                             0,         0, 0, 1);
        return Transform(m, Transpose(m));
    }
    Transform LookAt(Point3f pos, Point3f look, Vector3f up){
        //It is easiler to calculate camera_to_world transformation and then use Inverse() to calculate world_to_camera transformation
        SquareMatrix<4> world_to_camera;
        //First, we know that (0, 0, 0) in camera space is pos in world space
        world_to_camera[0][3] = pos.x;
        world_to_camera[1][3] = pos.y;
        world_to_camera[2][3] = pos.z;
        //Second, we know that z axis in camera space is normalize(look - pos) in world space
        Vector3f cameraz = Normalize(look - pos);
        world_to_camera[0][2] = cameraz.x;
        world_to_camera[1][2] = cameraz.y;
        world_to_camera[2][2] = cameraz.z;
        //Third, we know that y axis in camera space is up in world space
        up = Normalize(up);
        world_to_camera[0][1] = up.x;
        world_to_camera[1][1] = up.y;
        world_to_camera[2][1] = up.z;
        //Finally, x axis in camera space is the cross product of up and worldz
        Vector3f camerax = Cross(cameraz, up);
        world_to_camera[0][0] = camerax.x;
        world_to_camera[0][1] = camerax.y;
        world_to_camera[0][2] = camerax.z;

        return Transform(world_to_camera, InvertOrExit(world_to_camera));
    }



}