#include "core/bvh4.h"

#include <limits>
#include <utility>
#include <vector>

namespace gfx {

namespace {

// Half the surface area of a BVH2 node's box: used to decide which
// child is most worth opening up when filling a 4-wide node
float half_area(const BvhNode& node) {
    const Vec3 e = node.box_max - node.box_min;
    return e.x * e.y + e.y * e.z + e.z * e.x;
}

void set_empty(Bvh4Node& node, int i) {
    const float inf = std::numeric_limits<float>::infinity();

    // An inverted box: min > max on every axis, so the slab test fails
    node.min_x[i] = inf;
    node.min_y[i] = inf;
    node.min_z[i] = inf;
    node.max_x[i] = -inf;
    node.max_y[i] = -inf;
    node.max_z[i] = -inf;
    node.child[i] = kBvh4Empty;
    node.count[i] = 0;
}

} // namespace

Bvh4 build_bvh4(const Bvh& bvh) {
    Bvh4 out;

    if (bvh.nodes.empty()) {
        return out;
    }

    out.tri_indices = bvh.tri_indices;
    out.nodes.reserve(bvh.nodes.size() / 2 + 1);
    out.nodes.emplace_back();

    // (BVH2 node whose children this 4-wide node will hold, 4-wide node)
    std::vector<std::pair<std::uint32_t, std::uint32_t>> work;
    work.push_back({0, 0});

    while (!work.empty()) {
        const auto [source, target] = work.back();
        work.pop_back();

        // Gather up to four BVH2 nodes to become the children: start
        // with the source's children (or the source itself if it is a
        // leaf, which only happens for a root leaf) and keep replacing
        // the largest inner node by its two children while there is room
        std::uint32_t children[4];
        int count = 0;

        const BvhNode& src = bvh.nodes[source];

        if (src.count > 0) {
            children[count++] = source;
        } else {
            children[count++] = src.first_or_left;
            children[count++] = src.first_or_left + 1;
        }

        while (count < 4) {
            int best = -1;
            float best_area = -1.0f;

            for (int i = 0; i < count; ++i) {
                const BvhNode& n = bvh.nodes[children[i]];

                if (n.count == 0 && half_area(n) > best_area) {
                    best = i;
                    best_area = half_area(n);
                }
            }

            if (best < 0) {
                break;  // only leaves left
            }

            const std::uint32_t opened = children[best];
            children[best] = bvh.nodes[opened].first_or_left;
            children[count++] = bvh.nodes[opened].first_or_left + 1;
        }

        // Fill the node. Note: emplace_back below may move out.nodes,
        // so the node is written through its index each time
        for (int i = 0; i < 4; ++i) {
            if (i >= count) {
                set_empty(out.nodes[target], i);
                continue;
            }

            const BvhNode& c = bvh.nodes[children[i]];

            std::uint32_t child = 0;
            std::uint32_t triangles = 0;

            if (c.count > 0) {
                child = c.first_or_left;  // leaf: first triangle
                triangles = c.count;
            } else {
                child = static_cast<std::uint32_t>(out.nodes.size());
                out.nodes.emplace_back();
                work.push_back({children[i], child});
            }

            Bvh4Node& node = out.nodes[target];
            node.min_x[i] = c.box_min.x;
            node.min_y[i] = c.box_min.y;
            node.min_z[i] = c.box_min.z;
            node.max_x[i] = c.box_max.x;
            node.max_y[i] = c.box_max.y;
            node.max_z[i] = c.box_max.z;
            node.child[i] = child;
            node.count[i] = triangles;
        }
    }

    return out;
}

} // namespace gfx
