#include "core/brdf.h"
#include "check.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

using gfx::Vec3;

const double kPiD = 3.14159265358979323846;

// A tiny deterministic random number generator (the PCG hash of the CUDA
// code, chained). The standard library's distributions give different
// numbers on different compilers; this one gives the same everywhere
class Rng {
public:
    explicit Rng(std::uint32_t seed) : m_state(seed) {}

    // Uniform in [0, 1)
    float next() {
        m_state = m_state * 747796405u + 2891336453u;
        const std::uint32_t word =
            ((m_state >> ((m_state >> 28u) + 4u)) ^ m_state) * 277803737u;
        const std::uint32_t out = (word >> 22u) ^ word;

        return static_cast<float>(out >> 8) * (1.0f / 16777216.0f);
    }

private:
    std::uint32_t m_state;
};

bool close_to(double a, double b, double eps) {
    return std::fabs(a - b) <= eps;
}

// Relative difference of a and b, measured against b
double rel_diff(double a, double b) {
    return std::fabs(a - b) / std::fabs(b);
}

// A unit view vector with n.v = cos_v (the normal is +Z) at azimuth phi
Vec3 view_vector(float cos_v, float phi) {
    const float s = std::sqrt(std::fmax(0.0f, 1.0f - cos_v * cos_v));

    return {s * std::cos(phi), s * std::sin(phi), cos_v};
}

void test_constants() {
    static_assert(gfx::kDielectricF0 > 0.0f && gfx::kDielectricF0 < 0.1f,
                  "a dielectric reflects a few percent head on");
    static_assert(gfx::kGgxMinAlpha > 0.0f && gfx::kGgxMinAlpha < 0.01f,
                  "the alpha floor is a very sharp lobe, not a delta");
    static_assert(gfx::kSpecularProbabilityMin > 0.0f &&
                      gfx::kSpecularProbabilityMax < 1.0f &&
                      gfx::kSpecularProbabilityMin <
                          gfx::kSpecularProbabilityMax,
                  "both lobes keep a chance when both exist");
}

void test_alpha() {
    // alpha = roughness^2, kept inside the limits
    CHECK(close_to(static_cast<double>(gfx::ggx_alpha_from_roughness(0.5f)),
                   0.25, 1.0e-7));
    CHECK(close_to(static_cast<double>(gfx::ggx_alpha_from_roughness(1.0f)),
                   1.0, 1.0e-7));
    CHECK(gfx::ggx_alpha_from_roughness(0.0f) == gfx::kGgxMinAlpha);
    CHECK(gfx::ggx_alpha_from_roughness(0.01f) == gfx::kGgxMinAlpha);
    CHECK(gfx::ggx_alpha_from_roughness(3.0f) == 1.0f);
    CHECK(gfx::ggx_alpha_from_roughness(
              std::numeric_limits<float>::quiet_NaN()) ==
          gfx::kGgxMinAlpha);

    // Increasing in roughness
    float previous = 0.0f;
    bool increasing = true;

    for (int i = 1; i <= 100; ++i) {
        const float a =
            gfx::ggx_alpha_from_roughness(static_cast<float>(i) / 100.0f);
        increasing = increasing && a >= previous;
        previous = a;
    }

    CHECK(increasing);
}

// The integral of D(h) (n.h) over the hemisphere of microfacet normals is
// 1: the microfacets' areas projected onto the surface add up to the
// surface. In spherical coordinates (theta from the normal, phi around
// it), d omega = sin(theta) dtheta dphi, and D does not depend on phi,
// so the integral is 2 pi * integral of D(cos t) cos t sin t dt
void test_d_normalisation() {
    const float alphas[] = {0.05f, 0.2f, 0.5f, 1.0f};

    // Midpoint rule. A lobe of alpha 0.05 is a few hundredths of a radian
    // wide, so the step must be far smaller: 400000 steps over a quarter
    // circle is 4e-6 radians
    constexpr int kSteps = 400000;
    const double dt = kPiD * 0.5 / static_cast<double>(kSteps);

    for (const float alpha : alphas) {
        double sum = 0.0;

        for (int i = 0; i < kSteps; ++i) {
            const double t = (static_cast<double>(i) + 0.5) * dt;
            const float c = static_cast<float>(std::cos(t));

            sum += static_cast<double>(gfx::ggx_d(alpha, c)) *
                   static_cast<double>(c) * std::sin(t) * dt;
        }

        const double integral = 2.0 * kPiD * sum;

        CHECK(close_to(integral, 1.0, 0.01));
    }

    // Peak value 1 / (pi alpha^2) at n.h = 1; zero for a microfacet turned
    // away from the surface
    CHECK(close_to(static_cast<double>(gfx::ggx_d(0.5f, 1.0f)),
                   1.0 / (kPiD * 0.25), 1.0e-3));
    CHECK(gfx::ggx_d(0.5f, 0.0f) == 0.0f);
    CHECK(gfx::ggx_d(0.5f, -0.3f) == 0.0f);
}

void test_smith() {
    const float alphas[] = {0.002f, 0.05f, 0.3f, 0.7f, 1.0f};
    const float cosines[] = {1.0f,  0.8f, 0.5f,  0.3f,
                             0.1f,  0.03f, 0.005f, 1.0e-4f};

    for (const float alpha : alphas) {
        // Straight on nothing is hidden
        CHECK(close_to(static_cast<double>(gfx::smith_g1(alpha, 1.0f)), 1.0,
                       1.0e-6));
        CHECK(gfx::smith_g1(alpha, 0.0f) == 0.0f);
        CHECK(gfx::smith_g1(alpha, -0.5f) == 0.0f);

        float previous = 2.0f;
        bool decreasing = true;

        for (const float cv : cosines) {
            const float g1 = gfx::smith_g1(alpha, cv);

            // Strictly inside (0, 1]
            CHECK(g1 > 0.0f && g1 <= 1.0f + 1.0e-6f);

            // 1 / (1 + Lambda), the other way to write it
            CHECK(close_to(static_cast<double>(g1),
                           1.0 / (1.0 + static_cast<double>(
                                      gfx::smith_lambda(alpha, cv))),
                           1.0e-4));

            // Falls towards the horizon
            decreasing = decreasing && g1 <= previous + 1.0e-6f;
            previous = g1;

            for (const float cl : cosines) {
                const float g2 =
                    gfx::smith_g2_height_correlated(alpha, cl, cv);
                const float g2_swapped =
                    gfx::smith_g2_height_correlated(alpha, cv, cl);
                const float g1_l = gfx::smith_g1(alpha, cl);

                // Both directions together are never more visible than
                // either alone
                CHECK(g2 > 0.0f);
                CHECK(g2 <= g1 * (1.0f + 1.0e-5f));
                CHECK(g2 <= g1_l * (1.0f + 1.0e-5f));

                // Symmetric in l and v
                CHECK(close_to(static_cast<double>(g2),
                               static_cast<double>(g2_swapped), 1.0e-6));

                // The definition 1 / (1 + Lambda(v) + Lambda(l))
                const double lam = static_cast<double>(
                    gfx::smith_lambda(alpha, cv) +
                    gfx::smith_lambda(alpha, cl));
                CHECK(rel_diff(static_cast<double>(g2), 1.0 / (1.0 + lam)) <
                      1.0e-3);

                // G2 / G1(v) in one piece
                const float ratio = gfx::smith_g2_over_g1(alpha, cl, cv);
                CHECK(ratio > 0.0f && ratio <= 1.0f + 1.0e-5f);
                CHECK(rel_diff(static_cast<double>(ratio),
                               static_cast<double>(g2) /
                                   static_cast<double>(g1)) < 1.0e-3);

                // Vis = G2 / (4 (n.l) (n.v))
                const double vis =
                    static_cast<double>(gfx::ggx_visibility(alpha, cl, cv));
                const double want = static_cast<double>(g2) /
                                    (4.0 * static_cast<double>(cl) *
                                     static_cast<double>(cv));
                CHECK(rel_diff(vis, want) < 1.0e-3);
            }
        }

        CHECK(decreasing);
    }

    // At or below the horizon: nothing, and no NaN
    CHECK(gfx::smith_g2_height_correlated(0.3f, 0.0f, 0.5f) == 0.0f);
    CHECK(gfx::smith_g2_height_correlated(0.3f, 0.5f, -0.1f) == 0.0f);
    CHECK(gfx::ggx_visibility(0.3f, 0.0f, 0.5f) == 0.0f);
    CHECK(gfx::smith_g2_over_g1(0.3f, -0.1f, 0.5f) == 0.0f);

    // A tiny cosine does not blow up (the 0 / 0 the naive form has)
    CHECK(gfx::finite_float(gfx::ggx_visibility(0.3f, 1.0e-30f, 0.5f)));
    CHECK(gfx::finite_float(gfx::ggx_visibility(0.3f, 1.0e-30f, 1.0e-30f)));
}

void test_fresnel() {
    // Straight on: F0. Along the surface: 1 (every colour)
    const Vec3 f0{0.04f, 0.5f, 1.0f};
    const Vec3 head_on = gfx::fresnel_schlick(f0, 1.0f);
    const Vec3 glancing = gfx::fresnel_schlick(f0, 0.0f);

    CHECK(head_on.x == f0.x && head_on.y == f0.y && head_on.z == f0.z);
    CHECK(glancing.x == 1.0f && glancing.y == 1.0f && glancing.z == 1.0f);
    CHECK(gfx::fresnel_schlick(0.04f, 1.0f) == 0.04f);
    CHECK(gfx::fresnel_schlick(0.04f, 0.0f) == 1.0f);

    // The middle of the curve: 0.04 + 0.96 * 0.5^5
    CHECK(close_to(static_cast<double>(gfx::fresnel_schlick(0.04f, 0.5f)),
                   0.04 + 0.96 / 32.0, 1.0e-6));

    // Falls monotonically with the cosine, between F0 and 1
    float previous = 2.0f;
    bool decreasing = true;

    for (int i = 0; i <= 100; ++i) {
        const float c = static_cast<float>(i) / 100.0f;
        const float f = gfx::fresnel_schlick(0.04f, c);
        decreasing = decreasing && f <= previous + 1.0e-7f;
        decreasing = decreasing && f >= 0.04f - 1.0e-7f && f <= 1.0f;
        previous = f;
    }

    CHECK(decreasing);

    // Outside 0..1 the cosine is clamped (no NaN, no value beyond 1)
    CHECK(gfx::fresnel_schlick(0.04f, -0.2f) == 1.0f);
    CHECK(gfx::fresnel_schlick(0.04f, 1.5f) == 0.04f);

    // The dielectric layer: absent for f0 = 0 at every angle
    CHECK(gfx::fresnel_dielectric_layer(0.0f, 1.0f) == 0.0f);
    CHECK(gfx::fresnel_dielectric_layer(0.0f, 0.0f) == 0.0f);
    CHECK(gfx::fresnel_dielectric_layer(0.0f, 0.3f) == 0.0f);
    CHECK(gfx::fresnel_dielectric_layer(0.04f, 0.3f) ==
          gfx::fresnel_schlick(0.04f, 0.3f));
}

// The visible-normal sampler returns valid microfacet normals
void test_vndf_validity() {
    const float alphas[] = {0.002f, 0.1f, 0.3f, 0.6f, 1.0f};
    const float view_z[] = {1.0f, 0.7f, 0.3f, 0.1f, 0.02f};

    Rng rng(12345u);

    int bad_length = 0;
    int below_horizon = 0;
    int back_facing = 0;
    int bad_reflection = 0;

    for (const float alpha : alphas) {
        for (const float z : view_z) {
            for (int i = 0; i < 4000; ++i) {
                const float phi = 6.2831853f * rng.next();
                const Vec3 v = view_vector(z, phi);
                // (separate statements: the order in which the arguments
                // of a call are evaluated is not specified)
                const float u1 = rng.next();
                const float u2 = rng.next();
                const Vec3 h = gfx::ggx_sample_visible_normal_local(
                    alpha, v, u1, u2);

                if (!close_to(static_cast<double>(gfx::length(h)), 1.0,
                              1.0e-5)) {
                    ++bad_length;
                }

                if (!(h.z > 0.0f)) {
                    ++below_horizon;
                }

                if (!(gfx::dot(v, h) > 0.0f)) {
                    ++back_facing;
                }

                // The mirror direction in h is a unit vector and v and l
                // make the same angle with h
                const Vec3 l = gfx::reflect_about(v, h);

                if (!close_to(static_cast<double>(gfx::length(l)), 1.0,
                              1.0e-4) ||
                    !close_to(static_cast<double>(gfx::dot(l, h)),
                              static_cast<double>(gfx::dot(v, h)), 1.0e-4)) {
                    ++bad_reflection;
                }
            }
        }
    }

    CHECK(bad_length == 0);
    CHECK(below_horizon == 0);
    CHECK(back_facing == 0);
    CHECK(bad_reflection == 0);

    // World space: an arbitrary normal, the half vector is in the
    // hemisphere of both n and v
    int bad_world = 0;

    for (int i = 0; i < 20000; ++i) {
        const Vec3 raw{rng.next() - 0.5f, rng.next() - 0.5f,
                       rng.next() - 0.5f};
        const Vec3 n = gfx::normalize(raw);

        if (gfx::dot(n, n) == 0.0f) {
            continue;
        }

        // A view vector in the upper hemisphere of n
        Vec3 t;
        Vec3 b;
        gfx::orthonormal_basis(n, t, b);
        const float cz = 0.02f + 0.98f * rng.next();
        const float phi = 6.2831853f * rng.next();
        const float s = std::sqrt(1.0f - cz * cz);
        const Vec3 v = t * (s * std::cos(phi)) + b * (s * std::sin(phi)) +
                       n * cz;

        const float alpha = 0.05f + 0.95f * rng.next();
        const float u1 = rng.next();
        const float u2 = rng.next();
        const Vec3 h = gfx::ggx_sample_visible_normal(n, v, alpha, u1, u2);

        if (!(gfx::dot(h, n) > 0.0f) || !(gfx::dot(h, v) > 0.0f) ||
            !close_to(static_cast<double>(gfx::length(h)), 1.0, 1.0e-4)) {
            ++bad_world;
        }
    }

    CHECK(bad_world == 0);

    // A view vector below the surface (an interpolated normal facing
    // away) still gives a finite microfacet normal
    {
        const Vec3 n{0.0f, 0.0f, 1.0f};
        const Vec3 v = gfx::normalize(Vec3{1.0f, 0.0f, -0.05f});
        const Vec3 h = gfx::ggx_sample_visible_normal(n, v, 0.3f, 0.4f, 0.6f);

        CHECK(gfx::finite_vec3(h));
        CHECK(h.z > 0.0f);
    }
}

// The directional albedo of the specular lobe with F = 1:
//     E(v) = integral over l of f_s(l, v) (n.l) d omega_l
// found two independent ways. They must agree; if the visible-normal
// sampler or the weight G2 / G1 were wrong (a wrong pdf, a missing
// factor) they would not
void test_albedo_consistency() {
    const float alphas[] = {0.1f, 0.3f, 0.6f, 1.0f};
    const float cos_vs[] = {1.0f, 0.7f, 0.3f, 0.1f};

    // The uniform estimate uses a jittered grid of kGrid x kGrid cells
    // over the hemisphere (2 million samples); the visible-normal one
    // needs far fewer, its weights are bounded by 1
    constexpr int kGrid = 1415;
    constexpr int kVndfSamples = 400000;

    Rng rng(777u);

    for (const float alpha : alphas) {
        for (const float cv : cos_vs) {
            const Vec3 v = view_vector(cv, 0.7f);

            // (1) Uniform directions on the hemisphere: density
            // 1 / (2 pi), so the estimator is f_s (n.l) 2 pi. cos(theta)
            // is uniform in [0, 1] for a uniform hemisphere
            double uniform_sum = 0.0;

            for (int i = 0; i < kGrid; ++i) {
                for (int j = 0; j < kGrid; ++j) {
                    const float cz = (static_cast<float>(i) + rng.next()) /
                                     static_cast<float>(kGrid);
                    const float phi =
                        6.2831853f * (static_cast<float>(j) + rng.next()) /
                        static_cast<float>(kGrid);
                    const float s = std::sqrt(
                        std::fmax(0.0f, 1.0f - cz * cz));
                    const Vec3 l{s * std::cos(phi), s * std::sin(phi), cz};
                    const Vec3 h = gfx::normalize(l + v);

                    uniform_sum += static_cast<double>(
                        gfx::ggx_specular_white(alpha, l.z, v.z, h.z) * l.z);
                }
            }

            const double uniform_albedo = uniform_sum * 2.0 * kPiD /
                static_cast<double>(kGrid) / static_cast<double>(kGrid);

            // (2) Visible normals: the weight is G2 / G1(v) (F = 1); a
            // reflected direction below the horizon weighs 0
            double vndf_sum = 0.0;

            for (int i = 0; i < kVndfSamples; ++i) {
                const float u1 = rng.next();
                const float u2 = rng.next();
                const Vec3 h = gfx::ggx_sample_visible_normal_local(
                    alpha, v, u1, u2);
                const Vec3 l = gfx::reflect_about(v, h);

                if (l.z > 0.0f) {
                    vndf_sum += static_cast<double>(
                        gfx::smith_g2_over_g1(alpha, l.z, v.z));
                }
            }

            const double vndf_albedo =
                vndf_sum / static_cast<double>(kVndfSamples);

            // Same number, found two ways
            CHECK(rel_diff(vndf_albedo, uniform_albedo) < 0.015);

            // A white surface cannot reflect more than it receives. The
            // visible-normal weights are at most 1, so the bound is
            // exact there; the uniform estimate carries a little noise
            CHECK(vndf_albedo <= 1.0 + 1.0e-3);
            CHECK(uniform_albedo <= 1.0 + 1.5e-2);
            CHECK(vndf_albedo > 0.3);
        }
    }
}

// The lobe-selection probability
void test_lobe_probability() {
    const Vec3 zero{0.0f, 0.0f, 0.0f};

    // No specular lobe: always the diffuse one. No diffuse lobe: always
    // the specular one (exactly)
    CHECK(gfx::specular_lobe_probability(zero, Vec3{0.5f, 0.5f, 0.5f}) ==
          0.0f);
    CHECK(gfx::specular_lobe_probability(Vec3{0.9f, 0.8f, 0.7f}, zero) ==
          1.0f);
    CHECK(gfx::specular_lobe_probability(zero, zero) == 0.0f);

    // Both lobes: between the limits, rising with the specular share
    float previous = 0.0f;
    bool inside = true;
    bool rising = true;

    for (int i = 1; i <= 100; ++i) {
        const float w = static_cast<float>(i) / 100.0f;
        const float p = gfx::specular_lobe_probability(
            Vec3{w, w * 0.5f, 0.0f}, Vec3{0.6f, 0.6f, 0.6f});

        inside = inside && p >= gfx::kSpecularProbabilityMin &&
                 p <= gfx::kSpecularProbabilityMax;
        rising = rising && p >= previous;
        previous = p;
    }

    CHECK(inside);
    CHECK(rising);

    // Typical materials: plastic head on (F 0.04, Kd 0.8) mostly diffuse;
    // a metal (no diffuse) purely specular
    const float plastic = gfx::specular_lobe_probability(
        Vec3{0.04f, 0.04f, 0.04f}, Vec3{0.768f, 0.768f, 0.768f});
    CHECK(plastic == gfx::kSpecularProbabilityMin);

    // The share, where it is not clamped: w_s / (w_s + w_d)
    CHECK(close_to(static_cast<double>(gfx::specular_lobe_probability(
                       Vec3{0.3f, 0.1f, 0.1f}, Vec3{0.3f, 0.2f, 0.1f})),
                   0.5, 1.0e-6));
}

// The pieces put together: a rough check that the specular BRDF of a
// white surface reflects the right fraction, and the sampling weight of
// the diffuse + glossy mix never exceeds what is received
void test_specular_brdf() {
    // f_s = D Vis F, per channel
    const float alpha = 0.3f;
    const Vec3 f0{0.9f, 0.6f, 0.2f};
    const float nl = 0.6f;
    const float nv = 0.8f;
    const float nh = 0.95f;
    const float vh = 0.9f;

    const Vec3 f = gfx::ggx_specular_brdf(alpha, f0, nl, nv, nh, vh);
    const float dv = gfx::ggx_d(alpha, nh) * gfx::ggx_visibility(alpha, nl, nv);
    const Vec3 fr = gfx::fresnel_schlick(f0, vh);

    CHECK(f.x == fr.x * dv && f.y == fr.y * dv && f.z == fr.z * dv);
    CHECK(f.x > f.y && f.y > f.z);

    // Below the horizon: zero
    const Vec3 below = gfx::ggx_specular_brdf(alpha, f0, -0.1f, nv, nh, vh);
    CHECK(below.x == 0.0f && below.y == 0.0f && below.z == 0.0f);

    // Reciprocity: f_s(l, v) = f_s(v, l) (the Fresnel and D terms depend
    // only on h, which is the same, and the rest is symmetric)
    const Vec3 swapped = gfx::ggx_specular_brdf(alpha, f0, nv, nl, nh, vh);
    CHECK(close_to(static_cast<double>(f.x), static_cast<double>(swapped.x),
                   1.0e-5 * static_cast<double>(f.x)));
}


// The density of the visible-normal sampling. Two ways to get the share of
// the sampled directions that point above the surface must agree: count
// them (the sampler discards the ones that mirror v below the horizon),
// and add up the density over the hemisphere of directions l
void test_vndf_pdf() {
    const float alphas[] = {0.2f, 0.5f, 1.0f};
    const float cos_vs[] = {1.0f, 0.5f, 0.15f};
    const int kSamples = 200000;
    const int kGrid = 500;

    for (const float alpha : alphas) {
        for (const float cos_v : cos_vs) {
            const Vec3 v = view_vector(cos_v, 0.4f);

            // (1) the fraction of sampled normals whose mirror image of v
            // is above the surface
            Rng rng(1234u);
            int valid = 0;

            for (int i = 0; i < kSamples; ++i) {
                const float u1 = rng.next();
                const float u2 = rng.next();
                const Vec3 h =
                    gfx::ggx_sample_visible_normal_local(alpha, v, u1, u2);

                if (gfx::reflect_about(v, h).z > 0.0f) {
                    ++valid;
                }
            }

            const double fraction =
                static_cast<double>(valid) / static_cast<double>(kSamples);

            // (2) the integral of the density over the hemisphere of l,
            // midpoint rule on a (cos theta, phi) grid: a uniform grid
            // there is uniform in solid angle, each cell has
            // 2 pi / (grid * grid) steradians
            double integral = 0.0;

            for (int i = 0; i < kGrid; ++i) {
                const float z = (static_cast<float>(i) + 0.5f) /
                                static_cast<float>(kGrid);
                const float r = std::sqrt(1.0f - z * z);

                for (int j = 0; j < kGrid; ++j) {
                    const float phi = 6.2831853f *
                        (static_cast<float>(j) + 0.5f) /
                        static_cast<float>(kGrid);
                    const Vec3 l{r * std::cos(phi), r * std::sin(phi), z};
                    const Vec3 h = gfx::normalize(l + v);

                    integral += static_cast<double>(
                        gfx::ggx_vndf_pdf(alpha, z, cos_v, h.z));
                }
            }

            integral *= 2.0 * kPiD /
                        (static_cast<double>(kGrid) *
                         static_cast<double>(kGrid));

            CHECK(rel_diff(integral, fraction) < 0.015);

            // Below the horizon: no density
            CHECK(gfx::ggx_vndf_pdf(alpha, -0.1f, cos_v, 0.9f) == 0.0f);
        }
    }
}

// The mixture density: its two terms, and the single-lobe limits
void test_mixture_pdf() {
    const float alpha = 0.3f;
    const float cl = 0.6f;
    const float cv = 0.8f;
    const float ch = 0.95f;

    const double diffuse = static_cast<double>(cl) / kPiD;
    const double glossy =
        static_cast<double>(gfx::ggx_vndf_pdf(alpha, cl, cv, ch));

    CHECK(close_to(static_cast<double>(
                       gfx::bsdf_mixture_pdf(0.0f, alpha, cv, cl, ch)),
                   diffuse, 1.0e-6));
    CHECK(close_to(static_cast<double>(
                       gfx::bsdf_mixture_pdf(1.0f, alpha, cv, cl, ch)),
                   glossy, 1.0e-6 * glossy));
    CHECK(close_to(static_cast<double>(
                       gfx::bsdf_mixture_pdf(0.25f, alpha, cv, cl, ch)),
                   0.75 * diffuse + 0.25 * glossy, 1.0e-6 * glossy));
    CHECK(gfx::bsdf_mixture_pdf(0.5f, alpha, cv, -0.2f, ch) == 0.0f);
}

// ---------------------------------------------------------------------
// Multiple importance sampling of the sun
// ---------------------------------------------------------------------

// A surface with both lobes, seen from a fixed direction (the normal is
// +Z), for the check that the three ways of estimating the light it
// reflects from the sun agree
struct MisSurface {
    float alpha;
    Vec3 f0;
    float kd;
    Vec3 v;
    float cos_v;
    float p_spec;
};

// The BRDF Kd / pi + f_s(l, v) in the green channel
double mis_brdf(const MisSurface& s, Vec3 l) {
    const Vec3 h = gfx::normalize(l + s.v);
    const Vec3 f_s = gfx::ggx_specular_brdf(
        s.alpha, s.f0, l.z, s.cos_v, h.z, gfx::dot(s.v, h));

    return static_cast<double>(s.kd) / kPiD + static_cast<double>(f_s.y);
}

double mis_pdf(const MisSurface& s, Vec3 l) {
    const Vec3 h = gfx::normalize(l + s.v);

    return static_cast<double>(
        gfx::bsdf_mixture_pdf(s.p_spec, s.alpha, s.cos_v, l.z, h.z));
}

struct MeanAndError {
    double mean;
    double error;   // standard error of the mean
};

// Mean and standard error from the sums of kBatches equal batches
MeanAndError batch_statistics(const double* batch_sums, int batches,
                              int per_batch) {
    double mean = 0.0;

    for (int b = 0; b < batches; ++b) {
        mean += batch_sums[b] / static_cast<double>(per_batch);
    }

    mean /= static_cast<double>(batches);

    double var = 0.0;

    for (int b = 0; b < batches; ++b) {
        const double dev =
            batch_sums[b] / static_cast<double>(per_batch) - mean;

        var += dev * dev;
    }

    var /= static_cast<double>(batches - 1);

    return {mean, std::sqrt(var / static_cast<double>(batches))};
}

// The radiance a surface reflects towards the viewer from the sun (a cone
// of radiance L_sun = E / solid angle, E = 1), the integral of
// f(l) cos(l) L_sun over the cone, by four methods:
//  - a fine grid over the cone (the reference),
//  - (a) aiming at the cone, weight 1,
//  - (b) bouncing, counting the bounce rays that land in the cone,
//  - (c) both, with the power-heuristic weights of the path tracer.
// The estimates and their standard errors are returned in out.
struct MisResult {
    double reference;
    MeanAndError nee;
    MeanAndError bounce;
    MeanAndError mis;
};

MisResult run_mis_case(float alpha, float cos_v, float offset_degrees) {
    const int kBatches = 64;
    const int kPerBatch = 3000;
    const int kGrid = 300;

    MisSurface s;

    s.alpha = alpha;
    s.f0 = {0.9f, 0.7f, 0.4f};
    s.kd = 0.5f;
    s.v = view_vector(cos_v, 0.3f);
    s.cos_v = cos_v;
    s.p_spec = gfx::specular_lobe_probability(
        gfx::fresnel_schlick(s.f0, cos_v), Vec3{s.kd, s.kd, s.kd});

    // The cone: its axis is the mirror direction of v, tilted by the
    // offset (sideways, perpendicular to the mirror direction)
    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 mirror = gfx::reflect_about(s.v, normal);
    const Vec3 side = gfx::normalize(gfx::cross(normal, mirror));
    const float offset = offset_degrees * (gfx::kPi / 180.0f);
    const Vec3 axis = gfx::normalize(
        mirror * std::cos(offset) + side * std::sin(offset));

    const float cos_half = gfx::sun_cone_cos(gfx::kSunHalfAngle);
    const double omega =
        2.0 * kPiD * (1.0 - static_cast<double>(cos_half));
    const double l_sun = 1.0 / omega;
    const double pdf_sun = 1.0 / omega;

    MisResult r;

    // The reference: the average of f cos over the cone, E = 1
    double sum = 0.0;

    for (int i = 0; i < kGrid; ++i) {
        for (int j = 0; j < kGrid; ++j) {
            const Vec3 l = gfx::sample_sun_cone(
                axis, cos_half,
                (static_cast<float>(i) + 0.5f) / static_cast<float>(kGrid),
                (static_cast<float>(j) + 0.5f) / static_cast<float>(kGrid));

            if (l.z > 0.0f) {
                sum += mis_brdf(s, l) * static_cast<double>(l.z);
            }
        }
    }

    r.reference = sum / (static_cast<double>(kGrid) *
                         static_cast<double>(kGrid));

    double sums_a[kBatches] = {};
    double sums_b[kBatches] = {};
    double sums_c[kBatches] = {};
    Rng rng(777u);

    for (int b = 0; b < kBatches; ++b) {
        for (int i = 0; i < kPerBatch; ++i) {
            // (a) the aimed ray: f cos (E = 1), weight w_sun
            double est_a = 0.0;
            double est_a_weighted = 0.0;
            const Vec3 l = gfx::sample_sun_cone(
                axis, cos_half, rng.next(), rng.next());

            if (l.z > 0.0f) {
                est_a = mis_brdf(s, l) * static_cast<double>(l.z);

                const double ratio = mis_pdf(s, l) / pdf_sun;

                est_a_weighted = est_a / (1.0 + ratio * ratio);
            }

            // (b) the bounce ray: f cos L_sun / pdf if it is in the cone,
            // weight w_bsdf. The third random number is only drawn when
            // there is a choice, like in the path tracer
            const float u1 = rng.next();
            const float u2 = rng.next();
            const float u3 = (s.p_spec > 0.0f && s.p_spec < 1.0f)
                ? rng.next() : 0.0f;
            const gfx::BsdfSample bs = gfx::sample_bsdf_mixture(
                normal, s.v, s.alpha, s.p_spec, u1, u2, u3);

            double est_b = 0.0;
            double est_b_weighted = 0.0;

            if (bs.d.z > 0.0f && gfx::dot(bs.d, axis) >= cos_half) {
                const double pdf = mis_pdf(s, bs.d);

                if (pdf > 0.0) {
                    est_b = mis_brdf(s, bs.d) * static_cast<double>(bs.d.z) *
                            l_sun / pdf;

                    const double ratio = pdf_sun / pdf;

                    est_b_weighted = est_b / (1.0 + ratio * ratio);
                }
            }

            sums_a[b] += est_a;
            sums_b[b] += est_b;
            sums_c[b] += est_a_weighted + est_b_weighted;
        }
    }

    r.nee = batch_statistics(sums_a, kBatches, kPerBatch);
    r.bounce = batch_statistics(sums_b, kBatches, kPerBatch);
    r.mis = batch_statistics(sums_c, kBatches, kPerBatch);

    return r;
}

void test_mis_sun() {
    // alpha 0.005 is a roughness of about 0.07: the lobe is narrower than
    // the sun's disc, where the aimed ray alone is sparkly
    const float alphas[] = {0.005f, 0.03f, 0.1f, 0.4f};
    const float cos_vs[] = {0.9f, 0.4f};
    const float offsets[] = {0.0f, 2.0f};

    for (const float alpha : alphas) {
        for (const float cos_v : cos_vs) {
            for (const float offset : offsets) {
                const MisResult r = run_mis_case(alpha, cos_v, offset);

                // All three agree with the reference, the combination
                // tightly, the single strategies within what their own
                // noise says
                CHECK(rel_diff(r.mis.mean, r.reference) < 0.02);
                CHECK(std::fabs(r.nee.mean - r.reference) <
                      std::fmax(0.02 * r.reference, 4.0 * r.nee.error));
                CHECK(std::fabs(r.bounce.mean - r.reference) <
                      std::fmax(0.02 * r.reference, 4.0 * r.bounce.error));

                // The point of MIS: on the sharpest lobe the combination
                // is much less noisy than the aimed ray alone, and it is
                // never much worse than the better single strategy
                if (alpha < 0.01f && offset == 0.0f) {
                    CHECK(r.mis.error < r.nee.error);
                }

                CHECK(r.mis.error < 1.5 * std::fmin(r.nee.error,
                                                    r.bounce.error) +
                                        1.0e-3 * r.reference);
            }
        }
    }
}

} // namespace

int main() {
    test_constants();
    test_alpha();
    test_d_normalisation();
    test_smith();
    test_fresnel();
    test_vndf_validity();
    test_albedo_consistency();
    test_lobe_probability();
    test_specular_brdf();
    test_vndf_pdf();
    test_mixture_pdf();
    test_mis_sun();

    if (g_failures == 0) {
        std::printf("All brdf tests passed\n");
        return 0;
    }

    std::printf("%d brdf test failure(s)\n", g_failures);
    return 1;
}
