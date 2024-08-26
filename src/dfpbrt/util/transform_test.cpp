#include <ext/gtest/gtest.h>

#include <dfpbrt/dfpbrt.h>
#include <dfpbrt/util/rng.h>
#include <dfpbrt/util/transform.h>

using namespace dfpbrt;

TEST(RotateFromTo, Simple) {
    {
        // Same directions...
        Vector3f from(0, 0, 1), to(0, 0, 1);
        Transform r = RotateFromTo(from, to);
        Vector3f toNew = r(from);
        EXPECT_EQ(to, toNew);
    }

    {
        Vector3f from(0, 0, 1), to(1, 0, 0);
        Transform r = RotateFromTo(from, to);
        Vector3f toNew = r(from);
        EXPECT_EQ(to, toNew);
    }

    {
        Vector3f from(0, 0, 1), to(0, 1, 0);
        Transform r = RotateFromTo(from, to);
        Vector3f toNew = r(from);
        EXPECT_EQ(to, toNew);
    }
}

