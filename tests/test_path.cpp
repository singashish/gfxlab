#include "core/path_math.h"
#include "check.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

using gfx::Vec3;

const float kInfinity = std::numeric_limits<float>::infinity();
const float kNan = std::numeric_limits<float>::quiet_NaN();

bool close_to(float a, float b, float eps = 1.0e-5f) {
    return std::fabs(a - b) <= eps;
}

bool close_vec(Vec3 a, Vec3 b, float eps = 1.0e-5f) {
    return close_to(a.x, b.x, eps) && close_to(a.y, b.y, eps) &&
           close_to(a.z, b.z, eps);
}

bool same_vec(Vec3 a, Vec3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Compile-time sanity of the constants the renderer relies on
static_assert(gfx::kPtMaxBounces >= 1, "a path shades at least one surface");
static_assert(gfx::kPtSamplesPerFrame >= 1, "at least one sample per frame");
static_assert(gfx::kPtMaxSamples >= gfx::kPtSamplesPerFrame,
              "the cap is reachable");
static_assert(gfx::kSunIrradiance / gfx::kPi > 1.5f &&
                  gfx::kSunIrradiance / gfx::kPi < 1.7f,
              "a white surface facing the sun outputs about 1.6");
static_assert(gfx::kRouletteMinSurvival > 0.0f &&
                  gfx::kRouletteMaxSurvival < 1.0f,
              "roulette never kills or spares with certainty");

void test_srgb() {
    // End points and the standard's known mid value
    CHECK(gfx::decode_srgb(0.0f) == 0.0f);
    CHECK(close_to(gfx::decode_srgb(1.0f), 1.0f));
    CHECK(close_to(gfx::decode_srgb(0.5f), 0.21404f, 1.0e-4f));
    CHECK(gfx::encode_srgb(0.0f) == 0.0f);
    CHECK(close_to(gfx::encode_srgb(1.0f), 1.0f));
    CHECK(close_to(gfx::encode_srgb(0.21404f), 0.5f, 1.0e-4f));

    // The two pieces of the curve meet at the threshold
    const float below = gfx::decode_srgb(0.04045f - 1.0e-6f);
    const float above = gfx::decode_srgb(0.04045f + 1.0e-6f);
    CHECK(close_to(below, 0.0031308f, 1.0e-6f));
    CHECK(close_to(above, 0.0031308f, 1.0e-6f));
    CHECK(close_to(gfx::encode_srgb(0.0031308f), 0.04045f, 1.0e-5f));

    // The linear piece: 12.92 times
    CHECK(close_to(gfx::decode_srgb(0.02f), 0.02f / 12.92f, 1.0e-7f));

    // Round trip over the whole range, both ways
    for (int i = 0; i <= 100; ++i) {
        const float v = static_cast<float>(i) / 100.0f;
        CHECK(close_to(gfx::encode_srgb(gfx::decode_srgb(v)), v, 2.0e-5f));
        CHECK(close_to(gfx::decode_srgb(gfx::encode_srgb(v)), v, 2.0e-5f));
    }

    // decode is increasing
    float previous = -1.0f;
    bool increasing = true;

    for (int i = 0; i <= 255; ++i) {
        const float v = gfx::decode_srgb(static_cast<float>(i) / 255.0f);
        increasing = increasing && v > previous;
        previous = v;
    }

    CHECK(increasing);

    // Out of range values are clamped, so an albedo never exceeds 1
    CHECK(gfx::decode_srgb(2.0f) == gfx::decode_srgb(1.0f));
    CHECK(gfx::decode_srgb(-1.0f) == 0.0f);
    CHECK(gfx::encode_srgb(5.0f) == gfx::encode_srgb(1.0f));

    // Vec3 versions work per channel
    const Vec3 s{0.1f, 0.5f, 0.9f};
    const Vec3 l = gfx::decode_srgb(s);
    CHECK(l.x == gfx::decode_srgb(0.1f));
    CHECK(l.y == gfx::decode_srgb(0.5f));
    CHECK(l.z == gfx::decode_srgb(0.9f));
    CHECK(close_vec(gfx::encode_srgb(l), s, 2.0e-5f));
}

void test_tonemap() {
    CHECK(gfx::tonemap_aces(0.0f) == 0.0f);

    // The fit at x = 1: 2.54 / 3.16
    CHECK(close_to(gfx::tonemap_aces(1.0f), 2.54f / 3.16f, 1.0e-5f));

    // Monotone, inside 0..1, over a wide range
    float previous = -1.0f;
    bool increasing = true;
    bool in_range = true;

    for (int i = 0; i <= 4000; ++i) {
        const float x = static_cast<float>(i) * 0.005f;
        const float y = gfx::tonemap_aces(x);
        in_range = in_range && y >= 0.0f && y <= 1.0f;
        increasing = increasing && y >= previous;
        previous = y;
    }

    CHECK(in_range);
    CHECK(increasing);

    // Large input saturates to 1 without overflowing into NaN
    CHECK(close_to(gfx::tonemap_aces(100.0f), 1.0f));
    CHECK(gfx::tonemap_aces(1.0e6f) == 1.0f);
    CHECK(gfx::tonemap_aces(1.0e30f) == 1.0f);
    CHECK(gfx::tonemap_aces(3.0e38f) == 1.0f);

    // Negative input (cannot happen, but must be safe) gives 0
    CHECK(gfx::tonemap_aces(-1.0f) == 0.0f);

    // linear_to_display: exposure, tone mapping, sRGB
    const Vec3 mid{0.18f, 0.18f, 0.18f};
    const Vec3 shown = gfx::linear_to_display(mid, 1.0f);
    const float expected = gfx::encode_srgb(gfx::tonemap_aces(0.18f));
    CHECK(close_to(shown.x, expected));
    CHECK(close_to(shown.y, expected));

    const Vec3 doubled = gfx::linear_to_display(mid, 2.0f);
    CHECK(close_to(doubled.x,
                   gfx::encode_srgb(gfx::tonemap_aces(0.36f))));

    // NaN and infinity become black, never bright
    const float inf = kInfinity;
    const Vec3 black{0.0f, 0.0f, 0.0f};
    CHECK(same_vec(gfx::linear_to_display({kNan, 0.5f, 0.5f}, 1.0f), black));
    CHECK(same_vec(gfx::linear_to_display({0.5f, inf, 0.5f}, 1.0f), black));
    CHECK(same_vec(gfx::linear_to_display({0.5f, 0.5f, -inf}, 1.0f), black));

    // A very bright but finite value is white
    const Vec3 white = gfx::linear_to_display({5000.0f, 5000.0f, 5000.0f},
                                              1.0f);
    CHECK(close_to(white.x, 1.0f) && close_to(white.y, 1.0f) &&
          close_to(white.z, 1.0f));
}

void test_finite() {
    CHECK(gfx::finite_float(0.0f));
    CHECK(gfx::finite_float(-3.0e38f));
    CHECK(!gfx::finite_float(kNan));
    CHECK(!gfx::finite_float(kInfinity));
    CHECK(!gfx::finite_float(-kInfinity));
    CHECK(gfx::finite_vec3({1.0f, 2.0f, 3.0f}));
    CHECK(!gfx::finite_vec3({1.0f, kNan, 3.0f}));
    CHECK(!gfx::finite_vec3({1.0f, 2.0f, kInfinity}));

    const Vec3 product = gfx::modulate({1.0f, 2.0f, 3.0f},
                                       {0.5f, 0.25f, 2.0f});
    CHECK(same_vec(product, Vec3{0.5f, 0.5f, 6.0f}));
}

void test_sky() {
    const Vec3 zenith = gfx::sky_zenith_radiance();
    const Vec3 horizon = gfx::sky_horizon_radiance();
    const Vec3 ground = gfx::sky_ground_radiance();

    // The constants of the design (zenith and horizon scaled by 0.9)
    CHECK(close_vec(zenith, {0.27f, 0.45f, 0.855f}, 1.0e-6f));
    CHECK(close_vec(horizon, {0.765f, 0.792f, 0.855f}, 1.0e-6f));
    CHECK(close_vec(ground, {0.12f, 0.12f, 0.13f}, 1.0e-7f));

    // Straight up is exactly the zenith colour, the horizon exactly the
    // horizon colour, straight down exactly the ground colour
    CHECK(same_vec(gfx::sky_radiance({0.0f, 0.0f, 1.0f}), zenith));
    CHECK(same_vec(gfx::sky_radiance({1.0f, 0.0f, 0.0f}), horizon));
    CHECK(same_vec(gfx::sky_radiance({0.0f, 0.0f, -1.0f}), ground));

    // Zenith is bluer than the horizon: blue minus red is larger
    CHECK(zenith.z - zenith.x > horizon.z - horizon.x);

    // The sky does not depend on the azimuth
    const float z = 0.4f;
    const float r = std::sqrt(1.0f - z * z);
    const Vec3 a = gfx::sky_radiance({r, 0.0f, z});
    const Vec3 b = gfx::sky_radiance({0.0f, r, z});
    const Vec3 c = gfx::sky_radiance({-r * 0.6f, r * 0.8f, z});
    CHECK(close_vec(a, b, 1.0e-6f));
    CHECK(close_vec(a, c, 1.0e-6f));

    // Smooth and monotone above the horizon: red and green fall from the
    // horizon to the zenith, blue is the same at both ends
    bool red_falls = true;
    bool green_falls = true;
    Vec3 previous = gfx::sky_radiance({1.0f, 0.0f, 0.0f});

    for (int i = 1; i <= 200; ++i) {
        const float up = static_cast<float>(i) / 200.0f;
        const float side = std::sqrt(1.0f - up * up);
        const Vec3 now = gfx::sky_radiance({side, 0.0f, up});
        red_falls = red_falls && now.x <= previous.x;
        green_falls = green_falls && now.y <= previous.y;
        previous = now;
    }

    CHECK(red_falls);
    CHECK(green_falls);

    // The horizon colour weight is (1 - up)^3: at 30 degrees elevation
    // (up = 0.5) it is 0.125
    const Vec3 half = gfx::sky_radiance({std::sqrt(0.75f), 0.0f, 0.5f});
    CHECK(close_to(half.x, horizon.x * 0.125f + zenith.x * 0.875f, 1.0e-6f));

    // Continuous across the horizon: no jump between just above and just
    // below (the ground colour is faded in over the band)
    const Vec3 above = gfx::sky_radiance({1.0f, 0.0f, 1.0e-6f});
    const Vec3 below = gfx::sky_radiance({1.0f, 0.0f, -1.0e-6f});
    CHECK(close_vec(above, below, 1.0e-3f));

    // ... and the fade has no jump anywhere inside the band either: the
    // steepest change per step is bounded
    float largest_step = 0.0f;
    Vec3 last = gfx::sky_radiance({1.0f, 0.0f, 0.03f});

    for (int i = 1; i <= 600; ++i) {
        const float dz = 0.03f - static_cast<float>(i) * 0.0001f;
        const Vec3 now = gfx::sky_radiance({1.0f, 0.0f, dz});
        largest_step = std::fmax(largest_step, std::fabs(now.x - last.x));
        last = now;
    }

    CHECK(largest_step < 0.01f);

    // Below the band it is the plain ground colour; just below the
    // horizon it is still mostly sky (the band is thin but not a step)
    const Vec3 deep = gfx::sky_radiance({1.0f, 0.0f, -gfx::kHorizonBand});
    CHECK(same_vec(deep, ground));
    const Vec3 shallow = gfx::sky_radiance({1.0f, 0.0f, -0.001f});
    CHECK(shallow.x > ground.x + 0.4f);
    const Vec3 below_band = gfx::sky_radiance({1.0f, 0.0f, -0.5f});
    CHECK(same_vec(below_band, ground));

    // All channels positive and finite everywhere
    bool sane = true;

    for (int i = -100; i <= 100; ++i) {
        const float dz = static_cast<float>(i) / 100.0f;
        const float side = std::sqrt(1.0f - dz * dz);
        const Vec3 value = gfx::sky_radiance({side, 0.0f, dz});
        sane = sane && gfx::finite_vec3(value) && value.x > 0.0f &&
               value.y > 0.0f && value.z > 0.0f;
    }

    CHECK(sane);
}

void test_basis() {
    const Vec3 axes[] = {
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, -1.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},
        gfx::normalize({1.0e-4f, 0.0f, 1.0f}),
        gfx::normalize({0.0f, -1.0e-4f, 1.0f}),
        gfx::normalize({1.0e-4f, 1.0e-4f, -1.0f}),
        gfx::normalize({0.3f, -0.5f, 0.8f}),
        gfx::normalize({-0.7f, 0.2f, -0.1f}),
    };

    for (const Vec3& n : axes) {
        Vec3 t;
        Vec3 b;
        gfx::orthonormal_basis(n, t, b);

        CHECK(close_to(gfx::length(t), 1.0f, 1.0e-5f));
        CHECK(close_to(gfx::length(b), 1.0f, 1.0e-5f));
        CHECK(close_to(gfx::dot(t, b), 0.0f, 1.0e-5f));
        CHECK(close_to(gfx::dot(t, n), 0.0f, 1.0e-5f));
        CHECK(close_to(gfx::dot(b, n), 0.0f, 1.0e-5f));
    }
}

void test_cosine_hemisphere() {
    // A stratified grid of random numbers: every cell of the unit square
    // once, so the averages are deterministic and accurate
    constexpr int kSide = 64;
    const float kStep = 1.0f / static_cast<float>(kSide);

    const Vec3 normals[] = {
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, -1.0f},
        gfx::normalize({0.3f, -0.5f, 0.8f}),
    };

    for (const Vec3& n : normals) {
        double sum_cos = 0.0;
        bool above = true;
        bool unit = true;

        for (int i = 0; i < kSide; ++i) {
            for (int j = 0; j < kSide; ++j) {
                const float u1 = (static_cast<float>(i) + 0.5f) * kStep;
                const float u2 = (static_cast<float>(j) + 0.5f) * kStep;
                const Vec3 d = gfx::cosine_hemisphere(n, u1, u2);
                const float c = gfx::dot(d, n);

                above = above && c >= -1.0e-6f;
                unit = unit && close_to(gfx::length(d), 1.0f, 1.0e-5f);
                sum_cos += static_cast<double>(c);
            }
        }

        CHECK(above);
        CHECK(unit);

        // With density cos / pi the mean of cos is the integral of
        // cos^2 / pi over the hemisphere = 2/3
        const double mean = sum_cos / (kSide * kSide);
        CHECK(approx(mean, 2.0 / 3.0, 2.0e-3));
    }
}

void test_sun() {
    const float cos_half = gfx::sun_cone_cos(gfx::kSunHalfAngle);

    CHECK(close_to(cos_half, std::cos(gfx::to_radians(1.0f)), 1.0e-7f));
    CHECK(close_to(gfx::sun_cone_cos(0.0f), 1.0f, 1.0e-7f));

    // Solid angle: 2 pi (1 - cos). The sun's 1 degree cone
    const double exact = 2.0 * 3.14159265358979 *
                         (1.0 - std::cos(1.0 * 3.14159265358979 / 180.0));
    const float angle = gfx::sun_solid_angle(cos_half);
    CHECK(std::fabs(static_cast<double>(angle) - exact) < 1.0e-3 * exact);

    // Half the sphere and the whole sphere
    CHECK(close_to(gfx::sun_solid_angle(0.0f), 2.0f * gfx::kPi, 1.0e-5f));
    CHECK(close_to(gfx::sun_solid_angle(-1.0f), 4.0f * gfx::kPi, 1.0e-5f));
    CHECK(gfx::sun_solid_angle(1.0f) == 0.0f);

    // The disc's radiance times its solid angle is the irradiance
    const float radiance = gfx::sun_disc_radiance(gfx::kSunIrradiance,
                                                  cos_half);
    CHECK(close_to(radiance * angle, gfx::kSunIrradiance, 1.0e-3f));
    CHECK(radiance > 4000.0f && radiance < 6000.0f);

    // Sampling the cone, for the real sun and for a wide cone (the wide
    // one makes the statistics sharper)
    const float cones[] = {cos_half, std::cos(gfx::to_radians(30.0f))};

    const Vec3 axes[] = {
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, -1.0f},
        gfx::normalize({0.3f, -0.5f, 0.8f}),
        gfx::normalize({1.0e-4f, 0.0f, 1.0f}),
    };

    constexpr int kSide = 100;
    const float kStep = 1.0f / static_cast<float>(kSide);

    for (const float cone : cones) {
        for (const Vec3& axis : axes) {
            bool inside = true;
            bool unit = true;
            double sum_cos = 0.0;
            double sum_perp_x = 0.0;
            int inner = 0;

            // Half of the cone's solid angle: the angle with
            // 1 - cos = (1 - cone) / 2
            const float inner_cos = 1.0f - (1.0f - cone) * 0.5f;

            Vec3 t;
            Vec3 b;
            gfx::orthonormal_basis(axis, t, b);

            for (int i = 0; i < kSide; ++i) {
                for (int j = 0; j < kSide; ++j) {
                    const float u1 = (static_cast<float>(i) + 0.5f) * kStep;
                    const float u2 = (static_cast<float>(j) + 0.5f) * kStep;
                    const Vec3 d = gfx::sample_sun_cone(axis, cone, u1, u2);
                    const float c = gfx::dot(d, axis);

                    inside = inside && c >= cone - 1.0e-6f &&
                             c <= 1.0f + 1.0e-6f;
                    unit = unit && close_to(gfx::length(d), 1.0f, 1.0e-5f);
                    sum_cos += static_cast<double>(c);
                    sum_perp_x += static_cast<double>(gfx::dot(d, t));
                    inner += c >= inner_cos ? 1 : 0;
                }
            }

            const double count = static_cast<double>(kSide) * kSide;

            CHECK(inside);
            CHECK(unit);

            // Uniform in solid angle: cos(theta) is uniform between
            // cos_half and 1, so its mean is (1 + cos_half) / 2
            const double analytic = (1.0 + static_cast<double>(cone)) * 0.5;
            CHECK(approx(sum_cos / count, analytic, 1.0e-5));

            // ... the azimuth is uniform (no preferred side) ...
            CHECK(approx(sum_perp_x / count, 0.0, 1.0e-3));

            // ... and half the samples fall in the inner half of the
            // solid angle
            CHECK(approx(static_cast<double>(inner) / count, 0.5, 1.0e-2));
        }
    }

    // The ends of the random range: u1 = 0 is the axis, u1 -> 1 the rim
    const Vec3 axis{0.0f, 0.0f, 1.0f};
    CHECK(close_vec(gfx::sample_sun_cone(axis, cos_half, 0.0f, 0.3f), axis));
    const Vec3 rim = gfx::sample_sun_cone(axis, cos_half, 1.0f, 0.0f);
    CHECK(close_to(gfx::dot(rim, axis), cos_half, 1.0e-6f));
}

void test_roulette() {
    using gfx::russian_roulette_survival;

    // Bounds, whatever the throughput
    CHECK(russian_roulette_survival({0.0f, 0.0f, 0.0f}) ==
          gfx::kRouletteMinSurvival);
    CHECK(russian_roulette_survival({0.01f, 0.02f, 0.0f}) ==
          gfx::kRouletteMinSurvival);
    CHECK(russian_roulette_survival({1.0f, 1.0f, 1.0f}) ==
          gfx::kRouletteMaxSurvival);
    CHECK(russian_roulette_survival({50.0f, 0.0f, 0.0f}) ==
          gfx::kRouletteMaxSurvival);

    // In between it is the largest channel
    CHECK(close_to(russian_roulette_survival({0.5f, 0.2f, 0.1f}), 0.5f));
    CHECK(close_to(russian_roulette_survival({0.1f, 0.2f, 0.7f}), 0.7f));

    // Never a probability that would blow up the division by q
    bool in_bounds = true;
    bool non_decreasing = true;
    float previous = 0.0f;

    for (int i = 0; i <= 300; ++i) {
        const float v = static_cast<float>(i) * 0.01f;
        const float q = russian_roulette_survival({v, 0.0f, 0.0f});
        in_bounds = in_bounds && q >= gfx::kRouletteMinSurvival &&
                    q <= gfx::kRouletteMaxSurvival;
        non_decreasing = non_decreasing && q >= previous;
        previous = q;
    }

    CHECK(in_bounds);
    CHECK(non_decreasing);

    // Unbiased: survive with probability q and divide by q, the mean of
    // the throughput stays the same (checked over a deterministic set of
    // "random" numbers: exactly the first q of them survive)
    const float throughput = 0.3f;
    const float q = russian_roulette_survival({throughput, 0.0f, 0.0f});
    constexpr int kTrials = 10000;
    double sum = 0.0;

    for (int i = 0; i < kTrials; ++i) {
        const float u =
            (static_cast<float>(i) + 0.5f) / static_cast<float>(kTrials);

        if (u < q) {
            sum += static_cast<double>(throughput / q);
        }
    }

    CHECK(approx(sum / kTrials, static_cast<double>(throughput), 1.0e-3));
}

} // namespace

int main() {
    test_srgb();
    test_tonemap();
    test_finite();
    test_sky();
    test_basis();
    test_cosine_hemisphere();
    test_sun();
    test_roulette();

    if (g_failures == 0) {
        std::printf("All path tests passed\n");
        return 0;
    }

    std::printf("%d path test failure(s)\n", g_failures);
    return 1;
}
