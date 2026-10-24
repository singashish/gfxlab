#include "core/brdf.h"
#include "core/light_list.h"
#include "core/light_sampling.h"
#include "core/mat4.h"
#include "check.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using gfx::Vec3;

const double kPiD = 3.14159265358979323846;

// A tiny deterministic random number generator (the same as in
// test_brdf.cpp), so the numbers are identical on every compiler
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

double rel_diff(double a, double b) {
    return std::fabs(a - b) / std::fabs(b);
}

void test_area_scale() {
    // Identity and translation: 1
    CHECK(close_to(static_cast<double>(
                       gfx::area_scale(gfx::mat4_identity())), 1.0, 1.0e-6));
    CHECK(close_to(static_cast<double>(gfx::area_scale(
                       gfx::mat4_translate({3.0f, -2.0f, 7.0f}))),
                   1.0, 1.0e-6));

    // A uniform scale by s multiplies areas by s^2
    CHECK(close_to(static_cast<double>(gfx::area_scale(
                       gfx::mat4_scale({3.0f, 3.0f, 3.0f}))),
                   9.0, 1.0e-4));

    // A rotation does not change areas, also after a scale
    const gfx::Mat4 rot = gfx::mat4_rotate({0.0f, 0.0f, 1.0f}, 0.7f);
    const gfx::Mat4 both =
        rot * gfx::mat4_scale({0.5f, 0.5f, 0.5f});

    CHECK(close_to(static_cast<double>(gfx::area_scale(rot)), 1.0, 1.0e-5));
    CHECK(close_to(static_cast<double>(gfx::area_scale(both)), 0.25, 1.0e-5));

    // Non-uniform: |det|^(2/3) = (2 * 3 * 4)^(2/3), the geometric average
    const double expected = std::pow(24.0, 2.0 / 3.0);

    CHECK(close_to(static_cast<double>(gfx::area_scale(
                       gfx::mat4_scale({2.0f, 3.0f, 4.0f}))),
                   expected, 1.0e-3));

    // A mirror (negative determinant) still scales areas by a positive
    // number
    CHECK(close_to(static_cast<double>(gfx::area_scale(
                       gfx::mat4_scale({-1.0f, 1.0f, 1.0f}))),
                   1.0, 1.0e-6));
}

void test_triangle_area_normal() {
    float area = 0.0f;
    Vec3 normal;

    gfx::triangle_area_normal({0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f},
                              {0.0f, 3.0f, 0.0f}, area, normal);
    CHECK(close_to(static_cast<double>(area), 3.0, 1.0e-6));
    CHECK(close_to(static_cast<double>(normal.z), 1.0, 1.0e-6));

    // Degenerate: no area, zero normal
    gfx::triangle_area_normal({1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                              {2.0f, 1.0f, 1.0f}, area, normal);
    CHECK(area == 0.0f);
    CHECK(normal.x == 0.0f && normal.y == 0.0f && normal.z == 0.0f);
}

// Barycentric sampling is uniform by area
void test_barycentric_sampling() {
    Rng rng(99u);
    const int kSamples = 200000;
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    int inside = 0;

    // The sub-triangle with b0 > 0.5 (the corner triangle at corner 0,
    // scaled by 1/2) has 1/4 of the area
    for (int i = 0; i < kSamples; ++i) {
        float b0;
        float b1;
        float b2;

        gfx::sample_triangle_barycentric(rng.next(), rng.next(), b0, b1, b2);

        CHECK(b0 >= 0.0f && b1 >= 0.0f && b2 >= -1.0e-6f);
        CHECK(close_to(static_cast<double>(b0 + b1 + b2), 1.0, 1.0e-5));

        sum0 += static_cast<double>(b0);
        sum1 += static_cast<double>(b1);
        sum2 += static_cast<double>(b2);
        inside += b0 > 0.5f ? 1 : 0;
    }

    const double n = static_cast<double>(kSamples);

    CHECK(close_to(sum0 / n, 1.0 / 3.0, 0.004));
    CHECK(close_to(sum1 / n, 1.0 / 3.0, 0.004));
    CHECK(close_to(sum2 / n, 1.0 / 3.0, 0.004));
    CHECK(close_to(static_cast<double>(inside) / n, 0.25, 0.005));
}

void test_pick_from_cdf() {
    // Weights 1, 0 (zero weight), 3, 6 -> total 10
    const float cdf[] = {1.0f, 1.0f, 4.0f, 10.0f};
    const unsigned int count = 4u;
    const double weights[] = {0.1, 0.0, 0.3, 0.6};

    Rng rng(5u);
    const int kSamples = 200000;
    int hits[4] = {0, 0, 0, 0};

    for (int i = 0; i < kSamples; ++i) {
        float p = 0.0f;
        const unsigned int k = gfx::pick_from_cdf(cdf, count, rng.next(), p);

        CHECK(k < count);
        CHECK(close_to(static_cast<double>(p), weights[k], 1.0e-6));
        ++hits[k];
    }

    for (unsigned int k = 0; k < count; ++k) {
        const double freq =
            static_cast<double>(hits[k]) / static_cast<double>(kSamples);

        // Five sigma of a binomial frequency
        const double sigma = std::sqrt(weights[k] * (1.0 - weights[k]) /
                                       static_cast<double>(kSamples));

        CHECK(close_to(freq, weights[k], 5.0 * sigma + 1.0e-9));
        CHECK(close_to(static_cast<double>(
                           gfx::cdf_probability(cdf, count, k)),
                       weights[k], 1.0e-6));
    }

    CHECK(hits[1] == 0);

    // Edges: one entry, an empty list, the ends of the range
    const float single[] = {2.5f};
    float p = 0.0f;

    CHECK(gfx::pick_from_cdf(single, 1u, 0.0f, p) == 0u);
    CHECK(close_to(static_cast<double>(p), 1.0, 1.0e-6));
    CHECK(gfx::pick_from_cdf(single, 1u, 0.9999999f, p) == 0u);
    CHECK(gfx::pick_from_cdf(cdf, 0u, 0.5f, p) == 0u);
    CHECK(p == 0.0f);
    CHECK(gfx::pick_from_cdf(cdf, count, 0.9999999f, p) == 3u);
    CHECK(gfx::pick_from_cdf(cdf, count, 0.0f, p) == 0u);

    // The sorted-list search
    const std::uint32_t tris[] = {2u, 5u, 9u};

    CHECK(gfx::find_in_sorted(tris, 3u, 2u) == 0u);
    CHECK(gfx::find_in_sorted(tris, 3u, 9u) == 2u);
    CHECK(gfx::find_in_sorted(tris, 3u, 4u) == 3u);
    CHECK(gfx::find_in_sorted(tris, 3u, 10u) == 3u);
}

void test_area_to_solid_angle() {
    // A lamp at distance 2, seen head on: dA / d^2 steradians
    CHECK(close_to(static_cast<double>(
                       gfx::area_to_solid_angle_pdf(0.5f, 2.0f, 1.0f)),
                   2.0, 1.0e-6));

    // Seen at 60 degrees: twice as dense
    CHECK(close_to(static_cast<double>(
                       gfx::area_to_solid_angle_pdf(0.5f, 2.0f, 0.5f)),
                   4.0, 1.0e-6));
    CHECK(gfx::area_to_solid_angle_pdf(0.5f, 2.0f, 0.0f) == 0.0f);
}

// The list of lamp triangles: ascending, weights = area * luminance,
// zero-area and non-emissive triangles skipped
void test_build_light_list() {
    const std::vector<Vec3> positions = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f},
        {5.0f, 5.0f, 5.0f}};
    // triangles: 0 (area 1), 1 (degenerate), 2 (area 1), 3 (area 1)
    const std::vector<std::uint32_t> indices = {
        0, 1, 2,  0, 0, 1,  0, 1, 2,  0, 1, 2};
    const std::vector<float> luminance = {2.0f, 5.0f, 0.0f, 0.5f};

    const gfx::LightList list =
        gfx::build_light_list(positions, indices, luminance);

    CHECK(list.triangles.size() == 2u);
    CHECK(list.cdf.size() == 2u);

    if (list.triangles.size() == 2u) {
        CHECK(list.triangles[0] == 0u && list.triangles[1] == 3u);
        CHECK(close_to(static_cast<double>(list.cdf[0]), 2.0, 1.0e-6));
        CHECK(close_to(static_cast<double>(list.cdf[1]), 2.5, 1.0e-6));
    }

    CHECK(gfx::build_light_list(positions, indices, {0.0f, 0.0f, 0.0f, 0.0f})
              .cdf.empty());
}

// ---------------------------------------------------------------------
// MIS consistency for an area light
//
// A diffuse surface point at the origin (normal +Z, albedo a) under a
// disc lamp of radius R at height h directly above it, radiance Le on
// its lower face. The irradiance is E = pi Le R^2 / (R^2 + h^2), so
// the radiance leaving the point is L = a / pi * E = a Le R^2/(R^2+h^2).
// It is estimated by (a) aiming at the lamp (area sampling converted to
// solid angle), (b) bouncing with the cosine distribution and testing
// whether the ray reaches the disc, and (c) both, combined with the power
// heuristic exactly as the path tracer does
// ---------------------------------------------------------------------

struct StatPair {
    double mean;
    double error;   // standard error of the mean
};

StatPair batch_stats(const double* sums, int batches, int per_batch) {
    double mean = 0.0;

    for (int b = 0; b < batches; ++b) {
        mean += sums[b] / static_cast<double>(per_batch);
    }

    mean /= static_cast<double>(batches);

    double var = 0.0;

    for (int b = 0; b < batches; ++b) {
        const double dev = sums[b] / static_cast<double>(per_batch) - mean;

        var += dev * dev;
    }

    var /= static_cast<double>(batches - 1);

    return {mean, std::sqrt(var / static_cast<double>(batches))};
}

struct DiscResult {
    double analytic;
    StatPair light_only;
    StatPair bounce_only;
    StatPair mis;
};

DiscResult run_disc_case(float radius, float height, float albedo) {
    const int kBatches = 64;
    const int kPerBatch = 3000;
    const double le = 3.0;
    const double r2 = static_cast<double>(radius) * static_cast<double>(radius);
    const double h2 = static_cast<double>(height) * static_cast<double>(height);

    DiscResult res;

    res.analytic = static_cast<double>(albedo) * le * r2 / (r2 + h2);

    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 v = normal;  // a diffuse point: the view does not matter
    const float pdf_area = 1.0f / (gfx::kPi * radius * radius);
    const double f = static_cast<double>(albedo) / kPiD;

    double sums_a[kBatches] = {};
    double sums_b[kBatches] = {};
    double sums_c[kBatches] = {};
    Rng rng(4242u);

    for (int b = 0; b < kBatches; ++b) {
        for (int i = 0; i < kPerBatch; ++i) {
            // (a) a point of the disc, uniform by area
            const float r = radius * std::sqrt(rng.next());
            const float phi = 6.2831853f * rng.next();
            const Vec3 p{r * std::cos(phi), r * std::sin(phi), height};
            const float dist = gfx::length(p);
            const Vec3 l = p * (1.0f / dist);
            const float cos_light = l.z;  // the lamp normal is +Z
            const double pdf_light = static_cast<double>(
                gfx::area_to_solid_angle_pdf(pdf_area, dist, cos_light));
            const double pdf_bsdf_l = static_cast<double>(
                gfx::bsdf_mixture_pdf(0.0f, 1.0f, v.z, l.z, l.z));

            const double est_a = f * static_cast<double>(l.z) * le / pdf_light;
            const double w_light =
                1.0 / (1.0 + (pdf_bsdf_l / pdf_light) *
                                 (pdf_bsdf_l / pdf_light));

            // (b) a cosine-distributed bounce that reaches the disc
            const gfx::BsdfSample bs = gfx::sample_bsdf_mixture(
                normal, v, 1.0f, 0.0f, rng.next(), rng.next(), 0.0f);

            double est_b = 0.0;
            double est_b_weighted = 0.0;

            if (bs.d.z > 0.0f) {
                const float t = height / bs.d.z;
                const float hx = bs.d.x * t;
                const float hy = bs.d.y * t;

                if (hx * hx + hy * hy <= radius * radius) {
                    const double pdf_bsdf = static_cast<double>(
                        gfx::bsdf_mixture_pdf(0.0f, 1.0f, v.z, bs.d.z,
                                              bs.d.z));
                    const double pdf_l = static_cast<double>(
                        gfx::area_to_solid_angle_pdf(pdf_area, t * 1.0f,
                                                     bs.d.z));
                    // dist along the ray is t (the direction is unit)
                    est_b = f * static_cast<double>(bs.d.z) * le / pdf_bsdf;

                    const double ratio = pdf_l / pdf_bsdf;

                    est_b_weighted = est_b / (1.0 + ratio * ratio);
                }
            }

            sums_a[b] += est_a;
            sums_b[b] += est_b;
            sums_c[b] += est_a * w_light + est_b_weighted;
        }
    }

    res.light_only = batch_stats(sums_a, kBatches, kPerBatch);
    res.bounce_only = batch_stats(sums_b, kBatches, kPerBatch);
    res.mis = batch_stats(sums_c, kBatches, kPerBatch);

    return res;
}

void test_mis_area_light() {
    // radius, height: a small lamp far away (bouncing rarely finds it), a
    // big lamp close by (aiming at it is the worse one, since the
    // distance and angle vary a lot over its face), and a middle case
    const float radii[] = {0.1f, 3.0f, 1.0f};
    const float heights[] = {2.0f, 1.0f, 1.5f};

    for (int c = 0; c < 3; ++c) {
        const DiscResult r = run_disc_case(radii[c], heights[c], 0.7f);

        CHECK(rel_diff(r.mis.mean, r.analytic) < 0.02);
        CHECK(std::fabs(r.light_only.mean - r.analytic) <
              std::fmax(0.02 * r.analytic, 4.0 * r.light_only.error));
        CHECK(std::fabs(r.bounce_only.mean - r.analytic) <
              std::fmax(0.02 * r.analytic, 4.0 * r.bounce_only.error));

        // The combination is less noisy than the worse strategy, and
        // not much worse than the better one
        const double worse =
            std::fmax(r.light_only.error, r.bounce_only.error);
        const double better =
            std::fmin(r.light_only.error, r.bounce_only.error);

        if (c < 2) {
            CHECK(r.mis.error < worse);
        }

        CHECK(r.mis.error < 1.5 * better + 1.0e-3 * r.analytic);
    }
}

} // namespace

int main() {
    test_area_scale();
    test_triangle_area_normal();
    test_barycentric_sampling();
    test_pick_from_cdf();
    test_area_to_solid_angle();
    test_build_light_list();
    test_mis_area_light();

    if (g_failures == 0) {
        std::printf("All light tests passed\n");
        return 0;
    }

    std::printf("%d light test failure(s)\n", g_failures);
    return 1;
}
