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

    if (g_failures == 0) {
        std::printf("All brdf tests passed\n");
        return 0;
    }

    std::printf("%d brdf test failure(s)\n", g_failures);
    return 1;
}
