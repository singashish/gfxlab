#pragma once

/*
 * A 4-wide BVH: every node holds the boxes of up to four children.
 *
 * It is made from the binary BVH (build_bvh4 collapses it), so it uses
 * the same triangle order and needs no builder of its own. Each node
 * stores its children's boxes side by side ("structure of arrays"), so a
 * ray tests four boxes after one 128-byte read, and the tree has about
 * half as many levels as the binary one.
 *
 * Traversal is GFX_HD and works on raw pointers, like bvh_trace, so the
 * CPU tests and the CUDA ray tracer run the same code.
 */

#include "core/bvh.h"
#include "core/common.h"
#include "core/ray.h"
#include "core/vec3.h"

#include <cstdint>
#include <vector>

namespace gfx {

// 128 bytes. Slot i is
//   empty  when count[i] == 0 and child[i] == kBvh4Empty (its box is
//          inverted, so no ray ever hits it),
//   inner  when count[i] == 0: child[i] is a node index,
//   a leaf when count[i] > 0: triangles tri_indices[child[i] ..
//          child[i] + count[i])
struct Bvh4Node {
    float min_x[4];
    float min_y[4];
    float min_z[4];
    float max_x[4];
    float max_y[4];
    float max_z[4];
    std::uint32_t child[4];
    std::uint32_t count[4];
};

static_assert(sizeof(Bvh4Node) == 128, "Bvh4Node must stay 128 bytes");

constexpr std::uint32_t kBvh4Empty = 0xFFFFFFFFu;

struct Bvh4 {
    std::vector<Bvh4Node> nodes;
    std::vector<std::uint32_t> tri_indices;  // same order as the BVH2
};

// Collapses a binary BVH into a 4-wide one. An empty BVH gives an empty
// result
Bvh4 build_bvh4(const Bvh& bvh);

// A 4-wide node is never deeper than the binary node it came from, and
// each visit adds at most three pending children to the stack
constexpr int kBvh4StackSize = 3 * static_cast<int>(kBvhMaxDepth) + 4;

namespace bvh4_detail {

// Slab test of the ray against child slot i of a node
GFX_HD inline bool hits_child(const Bvh4Node& node,
                              int i,
                              Vec3 origin,
                              Vec3 inv_dir,
                              float t_max,
                              float& t_entry) {
    return ray_hits_box(origin, inv_dir,
                        Vec3{node.min_x[i], node.min_y[i], node.min_z[i]},
                        Vec3{node.max_x[i], node.max_y[i], node.max_z[i]},
                        t_max, t_entry);
}

} // namespace bvh4_detail

// Closest hit, like bvh_trace: leaf children are tested on the spot,
// inner children are pushed farthest first so the nearest pops next
GFX_HD inline bool bvh4_trace(const Bvh4Node* nodes,
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

    std::uint32_t stack[kBvh4StackSize];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        const Bvh4Node& node = nodes[stack[--top]];

        // Inner children the ray enters, with their entry distances
        std::uint32_t pending[4];
        float pending_t[4];
        int pending_count = 0;

        for (int i = 0; i < 4; ++i) {
            if (node.child[i] == kBvh4Empty) {
                continue;
            }

            float t_entry = 0.0f;

            if (!bvh4_detail::hits_child(node, i, origin, inv_dir,
                                         best_t, t_entry)) {
                continue;
            }

            if (node.count[i] > 0) {
                for (std::uint32_t k = 0; k < node.count[i]; ++k) {
                    const std::uint32_t tri =
                        tri_indices[node.child[i] + k];

                    float t, u, v;

                    if (ray_hits_triangle(
                            origin, dir,
                            positions[indices[tri * 3 + 0]],
                            positions[indices[tri * 3 + 1]],
                            positions[indices[tri * 3 + 2]],
                            t_min, best_t, t, u, v)) {
                        best_t = t;
                        hit.t = t;
                        hit.u = u;
                        hit.v = v;
                        hit.triangle = tri;
                        found = true;
                    }
                }
            } else {
                // Insertion sort, farthest first
                int j = pending_count++;

                while (j > 0 && pending_t[j - 1] < t_entry) {
                    pending[j] = pending[j - 1];
                    pending_t[j] = pending_t[j - 1];
                    --j;
                }

                pending[j] = node.child[i];
                pending_t[j] = t_entry;
            }
        }

        if (top + pending_count > kBvh4StackSize) {
            break;  // cannot happen for trees from build_bvh4
        }

        for (int i = 0; i < pending_count; ++i) {
            stack[top++] = pending[i];
        }
    }

    return found;
}

// Any hit, for shadow and AO rays: stops at the first triangle found
GFX_HD inline bool bvh4_occluded(const Bvh4Node* nodes,
                                 const std::uint32_t* tri_indices,
                                 const Vec3* positions,
                                 const std::uint32_t* indices,
                                 Vec3 origin,
                                 Vec3 dir,
                                 float t_min,
                                 float t_max) {
    if (nodes == nullptr) {
        return false;
    }

    const Vec3 inv_dir = ray_inverse_direction(dir);

    std::uint32_t stack[kBvh4StackSize];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        const Bvh4Node& node = nodes[stack[--top]];

        for (int i = 0; i < 4; ++i) {
            if (node.child[i] == kBvh4Empty) {
                continue;
            }

            float t_entry = 0.0f;

            if (!bvh4_detail::hits_child(node, i, origin, inv_dir,
                                         t_max, t_entry)) {
                continue;
            }

            if (node.count[i] > 0) {
                for (std::uint32_t k = 0; k < node.count[i]; ++k) {
                    const std::uint32_t tri =
                        tri_indices[node.child[i] + k];

                    float t, u, v;

                    if (ray_hits_triangle(
                            origin, dir,
                            positions[indices[tri * 3 + 0]],
                            positions[indices[tri * 3 + 1]],
                            positions[indices[tri * 3 + 2]],
                            t_min, t_max, t, u, v)) {
                        return true;
                    }
                }
            } else {
                if (top + 1 > kBvh4StackSize) {
                    return false;  // cannot happen for trees from build_bvh4
                }

                stack[top++] = node.child[i];
            }
        }
    }

    return false;
}

// Convenience wrappers for CPU code and tests
inline bool bvh4_trace(const Bvh4& bvh,
                       const Mesh& mesh,
                       Vec3 origin,
                       Vec3 dir,
                       float t_min,
                       float t_max,
                       BvhHit& hit) {
    if (bvh.nodes.empty()) {
        return false;
    }

    return bvh4_trace(bvh.nodes.data(), bvh.tri_indices.data(),
                      mesh.positions.data(), mesh.indices.data(),
                      origin, dir, t_min, t_max, hit);
}

inline bool bvh4_occluded(const Bvh4& bvh,
                          const Mesh& mesh,
                          Vec3 origin,
                          Vec3 dir,
                          float t_min,
                          float t_max) {
    if (bvh.nodes.empty()) {
        return false;
    }

    return bvh4_occluded(bvh.nodes.data(), bvh.tri_indices.data(),
                         mesh.positions.data(), mesh.indices.data(),
                         origin, dir, t_min, t_max);
}

} // namespace gfx
