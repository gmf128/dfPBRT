#include<dfpbrt/dfpbrt.h>
#include<dfpbrt/util/math.h>

namespace dfpbrt{
    template class SquareMatrix<2>;
    template class SquareMatrix<3>;
    template class SquareMatrix<4>;

    std::string SquareMatrix<3>::ToString() const{
        return std::format("{} {} {}\n {} {} {}\n{} {} {}\n", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
    }
}