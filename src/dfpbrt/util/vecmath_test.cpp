//
// Created by gmf on 2024/8/8.
//

// test1: namespace
#include <iostream>
#include "vecmath.h"
#include <ext/gtest/gtest.h>

template<typename S>
struct Testme
{
    /* data */
};

TEST(VectormathTest, Tuple2test_initialize){
    int i{};
    int j(0);
    EXPECT_EQ((i==j)&&(j==0), true);
}