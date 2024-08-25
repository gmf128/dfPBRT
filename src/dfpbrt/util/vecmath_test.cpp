//
// Created by gmf on 2024/8/8.
//

// test1: namespace
#include <iostream>
#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/vecmath.h>
#include <ext/gtest/gtest.h>
#include <cmath>

using namespace dfpbrt;

TEST(Vector2, Basics) {
    Vector2f vf(-1, 10);
    EXPECT_EQ(vf, Vector2f(Vector2i(-1, 10)));
    EXPECT_NE(vf, Vector2f(-1, 100));
    EXPECT_EQ(Vector2f(-2, 20), vf + vf);
    EXPECT_EQ(Vector2f(0, 0), vf - vf);
    EXPECT_EQ(Vector2f(-2, 20), vf * 2);
    EXPECT_EQ(Vector2f(-2, 20), 2 * vf);
    EXPECT_EQ(Vector2f(-0.5, 5), vf / 2);
    EXPECT_EQ(Vector2f(1, 10), Abs(vf));
    EXPECT_EQ(vf, Ceil(Vector2f(-1.5, 9.9)));
    EXPECT_EQ(vf, Floor(Vector2f(-.5, 10.01)));
    EXPECT_EQ(Vector2f(-20, 10), Min(vf, Vector2f(-20, 20)));
    EXPECT_EQ(Vector2f(-1, 20), Max(vf, Vector2f(-20, 20)));
    EXPECT_EQ(-1, MinComponentValue(vf));
    EXPECT_EQ(-10, MinComponentValue(-vf));
    EXPECT_EQ(10, MaxComponentValue(vf));
    EXPECT_EQ(1, MaxComponentValue(-vf));
    EXPECT_EQ(1, MaxComponentIndex(vf));
    EXPECT_EQ(0, MaxComponentIndex(-vf));
    EXPECT_EQ(0, MinComponentIndex(vf));
    EXPECT_EQ(1, MinComponentIndex(-vf));
    EXPECT_EQ(vf, Permute(vf, {0, 1}));
    EXPECT_EQ(Vector2f(10, -1), Permute(vf, {1, 0}));
    EXPECT_EQ(Vector2f(10, 10), Permute(vf, {1, 1}));
}

TEST(Vector3, Basics) {
    Vector3f vf(-1, 10, 2);
    EXPECT_EQ(vf, Vector3f(Vector3i(-1, 10, 2)));
    EXPECT_NE(vf, Vector3f(-1, 100, 2));
    EXPECT_EQ(Vector3f(-2, 20, 4), vf + vf);
    EXPECT_EQ(Vector3f(0, 0, 0), vf - vf);
    EXPECT_EQ(Vector3f(-2, 20, 4), vf * 2);
    EXPECT_EQ(Vector3f(-2, 20, 4), 2 * vf);
    EXPECT_EQ(Vector3f(-0.5, 5, 1), vf / 2);
    EXPECT_EQ(Vector3f(1, 10, 2), Abs(vf));
    EXPECT_EQ(vf, Ceil(Vector3f(-1.5, 9.9, 1.01)));
    EXPECT_EQ(vf, Floor(Vector3f(-.5, 10.01, 2.99)));
    EXPECT_EQ(Vector3f(-20, 10, 1.5), Min(vf, Vector3f(-20, 20, 1.5)));
    EXPECT_EQ(Vector3f(-1, 20, 2), Max(vf, Vector3f(-20, 20, 0)));
    EXPECT_EQ(-1, MinComponentValue(vf));
    EXPECT_EQ(-10, MinComponentValue(-vf));
    EXPECT_EQ(10, MaxComponentValue(vf));
    EXPECT_EQ(1, MaxComponentValue(-vf));
    EXPECT_EQ(1, MaxComponentIndex(vf));
    EXPECT_EQ(0, MaxComponentIndex(-vf));
    EXPECT_EQ(0, MinComponentIndex(vf));
    EXPECT_EQ(1, MinComponentIndex(-vf));
    EXPECT_EQ(vf, Permute(vf, {0, 1, 2}));
    EXPECT_EQ(Vector3f(10, -1, 2), Permute(vf, {1, 0, 2}));
    EXPECT_EQ(Vector3f(2, -1, 10), Permute(vf, {2, 0, 1}));
    EXPECT_EQ(Vector3f(10, 10, -1), Permute(vf, {1, 1, 0}));
}