#include <dfpbrt/util/float.h>
#include <ext/gtest/gtest.h>

#include <iostream>

int i = 5;
float s = 5.0;
    
TEST(FloatingPoint, IsNan){
    EXPECT_EQ(dfpbrt::IsNaN(s), false) << "float s = 5.0 should not be Nan";
    
};
    
   