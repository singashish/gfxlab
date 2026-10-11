#pragma once

#include <cmath>
#include <cstdio>

inline int g_failures = 0;

#define CHECK(cond)                                       \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL %s:%d  %s\n",      \
                         __FILE__, __LINE__, #cond);      \
            ++g_failures;                                 \
        }                                                 \
    } while (0)

inline bool approx(double a, double b, double eps = 1e-5) {
    return std::fabs(a - b) <= eps;
}
