// The tests deliberately use rays parallel to an axis (a zero direction
// component makes ray_inverse_direction return infinity, which the slab
// test relies on) and rays parallel to a triangle (rejected before any
// division). After inlining, MSVC's optimizer cannot see those guards
// and reports C4723 "potential divide by 0" in Release builds
#ifdef _MSC_VER
#pragma warning(disable : 4723)
#endif

#include "core/bvh.h"
#include "core/bvh4.h"
#include "core/mesh.h"
#include "core/obj_loader.h"
#include "core/ray.h"
#include "check.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using gfx::Vec3;

// Small deterministic random generator, so failures are repeatable
struct Lcg {
    std::uint32_t state = 12345u;

    float next01() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / 16777216.0f;
    }

    float range(float lo, float hi) {
        return lo + (hi - lo) * next01();
    }

    Vec3 point(float lo, float hi) {
        return {range(lo, hi), range(lo, hi), range(lo, hi)};
    }
};

// Reference answer: test every triangle
bool brute_force(const gfx::Mesh& mesh,
                 Vec3 origin,
                 Vec3 dir,
                 float& best_t) {
    bool found = false;
    best_t = 1.0e30f;

    for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
        float hit_t, u, v;

        if (gfx::ray_hits_triangle(
                origin, dir,
                mesh.positions[mesh.indices[t * 3 + 0]],
                mesh.positions[mesh.indices[t * 3 + 1]],
                mesh.positions[mesh.indices[t * 3 + 2]],
                0.0f, best_t, hit_t, u, v)) {
            best_t = hit_t;
            found = true;
        }
    }

    return found;
}

gfx::Mesh random_soup(std::uint32_t triangles, Lcg& rng) {
    gfx::Mesh mesh;

    for (std::uint32_t t = 0; t < triangles; ++t) {
        const Vec3 centre = rng.point(-10.0f, 10.0f);

        for (int k = 0; k < 3; ++k) {
            mesh.positions.push_back(centre + rng.point(-1.0f, 1.0f));
            mesh.indices.push_back(static_cast<std::uint32_t>(
                mesh.positions.size() - 1));
        }
    }

    return mesh;
}

bool box_contains(Vec3 outer_min, Vec3 outer_max,
                  Vec3 inner_min, Vec3 inner_max) {
    return inner_min.x >= outer_min.x && inner_min.y >= outer_min.y &&
           inner_min.z >= outer_min.z && inner_max.x <= outer_max.x &&
           inner_max.y <= outer_max.y && inner_max.z <= outer_max.z;
}

// Structural checks that hold for any correct BVH
void check_structure(const gfx::Bvh& bvh, const gfx::Mesh& mesh) {
    const std::size_t triangle_count = mesh.triangle_count();

    CHECK(bvh.tri_indices.size() == triangle_count);
    CHECK(bvh.nodes.size() <= triangle_count * 2);

    std::vector<int> seen(triangle_count, 0);

    for (const gfx::BvhNode& node : bvh.nodes) {
        if (node.count > 0) {
            // Every triangle of a leaf lies inside the leaf's box
            for (std::uint32_t i = 0; i < node.count; ++i) {
                const std::uint32_t tri =
                    bvh.tri_indices[node.first_or_left + i];

                ++seen[tri];

                for (int k = 0; k < 3; ++k) {
                    const Vec3 p = mesh.positions[mesh.indices[tri * 3 + k]];
                    CHECK(box_contains(node.box_min, node.box_max, p, p));
                }
            }
        } else {
            // Inner node: both children are inside it
            const gfx::BvhNode& l = bvh.nodes[node.first_or_left];
            const gfx::BvhNode& r = bvh.nodes[node.first_or_left + 1];

            CHECK(box_contains(node.box_min, node.box_max,
                               l.box_min, l.box_max));
            CHECK(box_contains(node.box_min, node.box_max,
                               r.box_min, r.box_max));
        }
    }

    // Every triangle belongs to exactly one leaf
    for (int count : seen) {
        CHECK(count == 1);
    }
}

// Structural checks for the 4-wide tree made from bvh
void check_structure4(const gfx::Bvh4& bvh4, const gfx::Mesh& mesh) {
    const std::size_t triangle_count = mesh.triangle_count();

    CHECK(bvh4.tri_indices.size() == triangle_count);

    std::vector<int> seen(triangle_count, 0);

    for (const gfx::Bvh4Node& node : bvh4.nodes) {
        int used = 0;

        for (int i = 0; i < 4; ++i) {
            if (node.child[i] == gfx::kBvh4Empty) {
                CHECK(node.count[i] == 0);
                continue;
            }

            ++used;

            const Vec3 lo{node.min_x[i], node.min_y[i], node.min_z[i]};
            const Vec3 hi{node.max_x[i], node.max_y[i], node.max_z[i]};

            if (node.count[i] > 0) {
                for (std::uint32_t k = 0; k < node.count[i]; ++k) {
                    const std::uint32_t tri =
                        bvh4.tri_indices[node.child[i] + k];
                    ++seen[tri];

                    for (int c = 0; c < 3; ++c) {
                        const Vec3 p =
                            mesh.positions[mesh.indices[tri * 3 + c]];
                        CHECK(box_contains(lo, hi, p, p));
                    }
                }
            } else {
                // Every slot of the inner child lies inside this box
                CHECK(node.child[i] < bvh4.nodes.size());
                const gfx::Bvh4Node& inner = bvh4.nodes[node.child[i]];

                for (int j = 0; j < 4; ++j) {
                    if (inner.child[j] == gfx::kBvh4Empty) {
                        continue;
                    }

                    CHECK(box_contains(
                        lo, hi,
                        {inner.min_x[j], inner.min_y[j], inner.min_z[j]},
                        {inner.max_x[j], inner.max_y[j], inner.max_z[j]}));
                }
            }
        }

        CHECK(used >= 1);
    }

    for (int count : seen) {
        CHECK(count == 1);
    }
}

// Random rays through the mesh's region: BVH and brute force must agree
void check_against_brute_force(const gfx::Bvh& bvh,
                               const gfx::Mesh& mesh,
                               float extent,
                               int rays,
                               Lcg& rng) {
    int mismatches = 0;
    int hits = 0;

    // The 4-wide tree must give exactly the same answers
    const gfx::Bvh4 bvh4 = gfx::build_bvh4(bvh);
    check_structure4(bvh4, mesh);

    for (int i = 0; i < rays; ++i) {
        const Vec3 origin = rng.point(-extent * 1.5f, extent * 1.5f);
        const Vec3 target = rng.point(-extent * 0.5f, extent * 0.5f);
        const Vec3 dir = gfx::normalize(target - origin);

        float brute_t = 0.0f;
        const bool brute_hit = brute_force(mesh, origin, dir, brute_t);

        gfx::BvhHit hit{};
        const bool bvh_hit = gfx::bvh_trace(
            bvh, mesh, origin, dir, 0.0f, 1.0e30f, hit);

        if (brute_hit != bvh_hit ||
            (brute_hit && std::fabs(brute_t - hit.t) >
                              1.0e-4f * std::max(1.0f, brute_t))) {
            ++mismatches;
        }

        // The any-hit query (shadow rays) must agree as well: blocked
        // with no distance limit exactly when there is a closest hit,
        // never blocked before the closest hit, blocked just after it
        if (gfx::bvh_occluded(bvh, mesh, origin, dir,
                              0.0f, 1.0e30f) != brute_hit) {
            ++mismatches;
        }

        gfx::BvhHit hit4{};
        const bool bvh4_hit = gfx::bvh4_trace(
            bvh4, mesh, origin, dir, 0.0f, 1.0e30f, hit4);

        if (brute_hit != bvh4_hit ||
            (brute_hit && std::fabs(brute_t - hit4.t) >
                              1.0e-4f * std::max(1.0f, brute_t))) {
            ++mismatches;
        }

        if (gfx::bvh4_occluded(bvh4, mesh, origin, dir,
                               0.0f, 1.0e30f) != brute_hit) {
            ++mismatches;
        }

        if (brute_hit) {
            if (gfx::bvh4_occluded(bvh4, mesh, origin, dir,
                                   0.0f, brute_t * 0.999f) ||
                !gfx::bvh4_occluded(bvh4, mesh, origin, dir,
                                    0.0f, brute_t * 1.001f)) {
                ++mismatches;
            }

            if (gfx::bvh_occluded(bvh, mesh, origin, dir,
                                  0.0f, brute_t * 0.999f)) {
                ++mismatches;
            }

            if (!gfx::bvh_occluded(bvh, mesh, origin, dir,
                                   0.0f, brute_t * 1.001f)) {
                ++mismatches;
            }
        }

        hits += brute_hit ? 1 : 0;
    }

    CHECK(mismatches == 0);

    // Make sure the test actually exercised hits as well as misses
    CHECK(hits > rays / 10);
    CHECK(hits < rays);
}

void test_ray_triangle() {
    const Vec3 a{0, 0, 0};
    const Vec3 b{1, 0, 0};
    const Vec3 c{0, 1, 0};

    float t, u, v;

    // Straight down onto the triangle from z = 2
    CHECK(gfx::ray_hits_triangle({0.25f, 0.25f, 2}, {0, 0, -1},
                                 a, b, c, 0.0f, 100.0f, t, u, v));
    CHECK(approx(t, 2.0));
    CHECK(approx(u, 0.25));
    CHECK(approx(v, 0.25));

    // From below: the test is two-sided
    CHECK(gfx::ray_hits_triangle({0.25f, 0.25f, -3}, {0, 0, 1},
                                 a, b, c, 0.0f, 100.0f, t, u, v));
    CHECK(approx(t, 3.0));

    // Outside the triangle
    CHECK(!gfx::ray_hits_triangle({0.75f, 0.75f, 2}, {0, 0, -1},
                                  a, b, c, 0.0f, 100.0f, t, u, v));

    // Parallel to the triangle's plane
    CHECK(!gfx::ray_hits_triangle({0.25f, 0.25f, 2}, {1, 0, 0},
                                  a, b, c, 0.0f, 100.0f, t, u, v));

    // Pointing away
    CHECK(!gfx::ray_hits_triangle({0.25f, 0.25f, 2}, {0, 0, 1},
                                  a, b, c, 0.0f, 100.0f, t, u, v));

    // Beyond t_max
    CHECK(!gfx::ray_hits_triangle({0.25f, 0.25f, 2}, {0, 0, -1},
                                  a, b, c, 0.0f, 1.5f, t, u, v));
}

void test_ray_box() {
    const Vec3 lo{-1, -1, -1};
    const Vec3 hi{1, 1, 1};
    float t_entry = 0.0f;

    const Vec3 dir{1, 0, 0};
    const Vec3 inv = gfx::ray_inverse_direction(dir);

    CHECK(gfx::ray_hits_box({-5, 0, 0}, inv, lo, hi, 100.0f, t_entry));
    CHECK(approx(t_entry, 4.0));

    // Origin inside the box: entry distance is 0
    CHECK(gfx::ray_hits_box({0, 0, 0}, inv, lo, hi, 100.0f, t_entry));
    CHECK(approx(t_entry, 0.0));

    // Misses above the box (a zero direction component gives infinities)
    CHECK(!gfx::ray_hits_box({-5, 0, 3}, inv, lo, hi, 100.0f, t_entry));

    // Box behind the ray
    CHECK(!gfx::ray_hits_box({5, 0, 0}, inv, lo, hi, 100.0f, t_entry));

    // Box farther than t_max
    CHECK(!gfx::ray_hits_box({-5, 0, 0}, inv, lo, hi, 3.0f, t_entry));
}

void test_empty_mesh() {
    gfx::Mesh mesh;
    const gfx::Bvh bvh = gfx::build_bvh(mesh);

    CHECK(bvh.nodes.empty());
    CHECK(bvh.tri_indices.empty());

    gfx::BvhHit hit{};
    CHECK(!gfx::bvh_trace(bvh, mesh, {0, 0, 0}, {0, 0, 1},
                          0.0f, 100.0f, hit));
}

void test_single_triangle() {
    gfx::Mesh mesh;
    mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.indices = {0, 1, 2};

    const gfx::Bvh bvh = gfx::build_bvh(mesh);

    CHECK(bvh.nodes.size() == 1);
    CHECK(bvh.nodes[0].count == 1);

    // A root that is a leaf becomes one 4-wide node with one leaf slot
    const gfx::Bvh4 bvh4 = gfx::build_bvh4(bvh);
    CHECK(bvh4.nodes.size() == 1);
    CHECK(bvh4.nodes[0].count[0] == 1);
    CHECK(bvh4.nodes[0].child[1] == gfx::kBvh4Empty);

    gfx::BvhHit hit4{};
    CHECK(gfx::bvh4_trace(bvh4, mesh, {0.2f, 0.2f, 5}, {0, 0, -1},
                          0.0f, 100.0f, hit4));
    CHECK(approx(hit4.t, 5.0));

    gfx::BvhHit hit{};
    CHECK(gfx::bvh_trace(bvh, mesh, {0.2f, 0.2f, 5}, {0, 0, -1},
                         0.0f, 100.0f, hit));
    CHECK(approx(hit.t, 5.0));
    CHECK(hit.triangle == 0);
}

void test_random_soup() {
    Lcg rng;
    const gfx::Mesh mesh = random_soup(2000, rng);
    const gfx::Bvh bvh = gfx::build_bvh(mesh);

    check_structure(bvh, mesh);

    const gfx::BvhStats stats = gfx::bvh_stats(bvh);
    CHECK(stats.leaf_count > 2000 / 8);
    CHECK(stats.max_depth <= 48);

    check_against_brute_force(bvh, mesh, 10.0f, 3000, rng);
}

void test_suzanne() {
    gfx::Mesh mesh;
    std::string error;

    const std::string path =
        std::string(GFX_ASSETS_DIR) + "/suzanne_s.obj";

    CHECK(gfx::load_obj(path, mesh, error));

    if (mesh.triangle_count() == 0) {
        return;
    }

    const gfx::Bvh bvh = gfx::build_bvh(mesh);
    check_structure(bvh, mesh);

    const gfx::BvhStats stats = gfx::bvh_stats(bvh);

    std::printf(
        "suzanne_s.obj: %zu triangles, %u nodes, %u leaves, "
        "max depth %u, largest leaf %u, %zu 4-wide nodes\n",
        mesh.triangle_count(), stats.node_count, stats.leaf_count,
        stats.max_depth, stats.max_leaf_size,
        gfx::build_bvh4(bvh).nodes.size());

    Lcg rng;
    rng.state = 777u;
    check_against_brute_force(bvh, mesh, 1.5f, 1000, rng);
}

} // namespace

int main() {
    test_ray_triangle();
    test_ray_box();
    test_empty_mesh();
    test_single_triangle();
    test_random_soup();
    test_suzanne();

    if (g_failures == 0) {
        std::printf("All BVH tests passed\n");
        return 0;
    }

    std::printf("%d BVH test failure(s)\n", g_failures);
    return 1;
}
