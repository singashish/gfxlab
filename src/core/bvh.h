#pragma once

/*
 * Bounding volume hierarchy over a mesh's triangles.
 *
 * Built on the CPU (build_bvh) and stored as two flat arrays so it can be
 * copied to the GPU unchanged:
 *   nodes        - a binary tree; node 0 is the root
 *   tri_indices  - triangle numbers, reordered so every leaf owns one
 *                  contiguous range of this array
 *
 * bvh_trace() walks the tree with raw pointers and is GFX_HD, so the CPU
 * tests and the CUDA ray tracer run exactly the same code.
 */

#include "core/common.h"
#include "core/mesh.h"
#include "core/ray.h"
#include "core/vec3.h"

#include <cstdint>
#include <vector>

namespace gfx {

// 32 bytes. A node with count > 0 is a leaf whose triangles are
// tri_indices[first_or_left .. first_or_left + count). A node with
// count == 0 is an inner node whose children are first_or_left and
// first_or_left + 1 (children are always stored next to each other)
struct BvhNode {
    Vec3 box_min;
    std::uint32_t first_or_left;
    Vec3 box_max;
    std::uint32_t count;
};

static_assert(sizeof(BvhNode) == 32, "BvhNode must stay 32 bytes");

struct Bvh {
    std::vector<BvhNode> nodes;
    std::vector<std::uint32_t> tri_indices;
};

struct BvhStats {
    std::uint32_t node_count = 0;
    std::uint32_t leaf_count = 0;
    std::uint32_t max_depth = 0;
    std::uint32_t max_leaf_size = 0;
};

// Builds the tree with the binned surface area heuristic (SAH).
// An empty mesh gives an empty BVH
Bvh build_bvh(const Mesh& mesh);

BvhStats bvh_stats(const Bvh& bvh);

struct BvhHit {
    float t;
    float u;
    float v;
    std::uint32_t triangle;  // triangle number in the mesh (index / 3)
};

// Deep enough for any tree the builder makes: its depth is limited by
// kBvhMaxDepth in bvh.cpp
constexpr int kBvhStackSize = 64;

// Closest hit along origin + t * dir for t in (t_min, t_max).
// Visits the nearer child first and skips boxes farther than the best hit
GFX_HD inline bool bvh_trace(const BvhNode* nodes,
                             const std::uint32_t* tri_indices,
                             const Vec3* positions,
                             const std::uint32_t* indices,
                             Vec3 origin,
                             Vec3 dir,
                             float t_min,
                             float t_max,
                             BvhHit& hit) {
    if (nodes == nullptr) {
        return false;
    }

    const Vec3 inv_dir = ray_inverse_direction(dir);

    float best_t = t_max;
    bool found = false;

    float t_entry = 0.0f;

    if (!ray_hits_box(origin, inv_dir,
                      nodes[0].box_min, nodes[0].box_max,
                      best_t, t_entry)) {
        return false;
    }

    std::uint32_t stack[kBvhStackSize];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        const BvhNode& node = nodes[stack[--top]];

        if (node.count > 0) {
            // Leaf: test its triangles
            for (std::uint32_t i = 0; i < node.count; ++i) {
                const std::uint32_t tri =
                    tri_indices[node.first_or_left + i];

                const Vec3 a = positions[indices[tri * 3 + 0]];
                const Vec3 b = positions[indices[tri * 3 + 1]];
                const Vec3 c = positions[indices[tri * 3 + 2]];

                float t, u, v;

                if (ray_hits_triangle(origin, dir, a, b, c,
                                      t_min, best_t, t, u, v)) {
                    best_t = t;
                    hit.t = t;
                    hit.u = u;
                    hit.v = v;
                    hit.triangle = tri;
                    found = true;
                }
            }

            continue;
        }

        // Inner node: test both children, then visit the nearer first
        const std::uint32_t left = node.first_or_left;
        const std::uint32_t right = left + 1;

        float t_left = 0.0f;
        float t_right = 0.0f;

        const bool hit_left = ray_hits_box(
            origin, inv_dir,
            nodes[left].box_min, nodes[left].box_max,
            best_t, t_left);

        const bool hit_right = ray_hits_box(
            origin, inv_dir,
            nodes[right].box_min, nodes[right].box_max,
            best_t, t_right);

        if (hit_left && hit_right) {
            // Push the farther child first so the nearer one pops first
            if (top + 2 > kBvhStackSize) {
                break;  // cannot happen for trees from build_bvh
            }

            if (t_left <= t_right) {
                stack[top++] = right;
                stack[top++] = left;
            } else {
                stack[top++] = left;
                stack[top++] = right;
            }
        } else if (hit_left || hit_right) {
            if (top + 1 > kBvhStackSize) {
                break;
            }

            stack[top++] = hit_left ? left : right;
        }
    }

    return found;
}

// Convenience wrapper for CPU code and tests
inline bool bvh_trace(const Bvh& bvh,
                      const Mesh& mesh,
                      Vec3 origin,
                      Vec3 dir,
                      float t_min,
                      float t_max,
                      BvhHit& hit) {
    if (bvh.nodes.empty()) {
        return false;
    }

    return bvh_trace(bvh.nodes.data(), bvh.tri_indices.data(),
                     mesh.positions.data(), mesh.indices.data(),
                     origin, dir, t_min, t_max, hit);
}

} // namespace gfx
