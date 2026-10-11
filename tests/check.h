#pragma once

#include <cmath>
#include <cstdio>

static int g_failures = 0;

#define CHECK(cond)                                       \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL %s:%d  %s\n",      \
                         __FILE__, __LINE__, #cond);      \
            ++g_failures;                                 \
        }                                                 \
    } while (0)

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}
