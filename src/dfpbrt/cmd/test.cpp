#include <dfpbrt/dfpbrt.h>
#include <ext/gtest/gtest.h>
#include <dfpbrt/util/log.h>

#include <dfpbrt/dfpbrt.h>

int main(int argc, char ** argv){
    testing::InitGoogleTest(&argc, argv);
    //Init()
    dfpbrt::InitDFPBRT();

    return RUN_ALL_TESTS();
}