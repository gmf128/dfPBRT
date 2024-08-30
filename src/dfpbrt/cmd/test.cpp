#include <dfpbrt/dfpbrt.h>
#include <ext/gtest/gtest.h>
#include <dfpbrt/util/log.h>

#include <dfpbrt/dfpbrt.h>

int main(int argc, char ** argv){
    testing::InitGoogleTest(&argc, argv);
    //Init()
    // Now, we will set all the options as default value
    DFPBRTOptions opt = new DFPBRTOptions();
    dfpbrt::InitDFPBRT(opt);

    return RUN_ALL_TESTS();
}