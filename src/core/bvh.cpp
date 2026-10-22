#include "core/bvh.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace gfx {

namespace {

// Surface area heuristic settings
constexpr int kBins = 16;                 // candidate split planes per axis
constexpr std::uint32_t kMaxLeafSize = 4; // force a split above this,
                                          // unless at kBvhMaxDepth (bvh.h)

// Traversal visits at most one pending sibling per level, so the stack
// in bvh_trace needs depth + 1 entries
static_assert(kBvhMaxDepth + 1 <= static_cast<std::uint32_t>(kBvhStackSize),
              "bvh_trace stack is too small for the deepest tree");

struct Box {
    Vec3 min;
    Vec3 max;
};

Box empty_box() {
    const float inf = std::numeric_limits<float>::infinity();
    return {{inf, inf, inf}, {-inf, -inf, -inf}};
}

void grow(Box& box, Vec3 p) {
    box.min = {std::min(box.min.x, p.x),
               std::min(box.min.y, p.y),
               std::min(box.min.z, p.z)};
    box.max = {std::max(box.max.x, p.x),
               std::max(box.max.y, p.y),
               std::max(box.max.z, p.z)};
}

// Union of two boxes. Taking min of mins and max of maxes (rather than
// growing by the other box's corners) keeps an empty box neutral:
// its +inf / -inf corners never leak into the result
void grow(Box& box, const Box& other) {
    box.min = {std::min(box.min.x, other.min.x),
               std::min(box.min.y, other.min.y),
               std::min(box.min.z, other.min.z)};
    box.max = {std::max(box.max.x, other.max.x),
               std::max(box.max.y, other.max.y),
               std::max(box.max.z, other.max.z)};
}

// Half the surface area; the factor 2 cancels out in every comparison
float half_area(const Box& box) {
    const Vec3 e = box.max - box.min;

    if (e.x < 0.0f || e.y < 0.0f || e.z < 0.0f) {
        return 0.0f;  // empty box
    }

    return e.x * e.y + e.y * e.z + e.z * e.x;
}

float axis_value(Vec3 v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

struct TriangleInfo {
    Box box;
    Vec3 centroid;
};

struct Bin {
    Box box = empty_box();
    std::uint32_t count = 0;
};

struct Split {
    int axis = -1;
    int bin = 0;        // triangles in bins [0, bin] go left
    float cost = std::numeric_limits<float>::infinity();
};

// Finds the cheapest split of tri_indices[first .. first + count)
Split find_split(const std::vector<TriangleInfo>& info,
                 const std::vector<std::uint32_t>& tri_indices,
                 std::uint32_t first,
                 std::uint32_t count,
                 const Box& centroid_box) {
    Split best;

    for (int axis = 0; axis < 3; ++axis) {
        const float lo = axis_value(centroid_box.min, axis);
        const float hi = axis_value(centroid_box.max, axis);

        // Every centroid at the same coordinate: no plane separates them
        if (!(hi > lo)) {
            continue;
        }

        const float scale = static_cast<float>(kBins) / (hi - lo);

        Bin bins[kBins];

        for (std::uint32_t i = 0; i < count; ++i) {
            const TriangleInfo& t = info[tri_indices[first + i]];

            const int b = std::min(
                kBins - 1,
                static_cast<int>(
                    (axis_value(t.centroid, axis) - lo) * scale));

            grow(bins[b].box, t.box);
            ++bins[b].count;
        }

        // Sweep from the left and from the right to get the area and
        // count of each side for every plane between two bins
        float left_area[kBins - 1];
        std::uint32_t left_count[kBins - 1];

        Box box = empty_box();
        std::uint32_t n = 0;

        for (int i = 0; i < kBins - 1; ++i) {
            grow(box, bins[i].box);
            n += bins[i].count;
            left_area[i] = half_area(box);
            left_count[i] = n;
        }

        box = empty_box();
        n = 0;

        for (int i = kBins - 1; i > 0; --i) {
            grow(box, bins[i].box);
            n += bins[i].count;

            const float cost =
                static_cast<float>(left_count[i - 1]) * left_area[i - 1] +
                static_cast<float>(n) * half_area(box);

            if (left_count[i - 1] > 0 && n > 0 && cost < best.cost) {
                best.axis = axis;
                best.bin = i - 1;
                best.cost = cost;
            }
        }
    }

    return best;
}

} // namespace

Bvh build_bvh(const Mesh& mesh) {
    Bvh bvh;

    const std::uint32_t triangle_count =
        static_cast<std::uint32_t>(mesh.indices.size() / 3);

    if (triangle_count == 0) {
        return bvh;
    }

    // Bounds and centroid of every triangle, computed once
    std::vector<TriangleInfo> info(triangle_count);

    for (std::uint32_t t = 0; t < triangle_count; ++t) {
        Box box = empty_box();

        for (int k = 0; k < 3; ++k) {
            grow(box, mesh.positions[mesh.indices[t * 3 + k]]);
        }

        info[t].box = box;
        info[t].centroid = (box.min + box.max) * 0.5f;
    }

    bvh.tri_indices.resize(triangle_count);

    for (std::uint32_t t = 0; t < triangle_count; ++t) {
        bvh.tri_indices[t] = t;
    }

    // A binary tree with n leaves has 2n - 1 nodes, and every leaf holds
    // at least one triangle
    bvh.nodes.reserve(static_cast<std::size_t>(triangle_count) * 2);
    bvh.nodes.push_back({{}, 0, {}, triangle_count});

    // Iterative build: (node index, depth)
    std::vector<std::pair<std::uint32_t, std::uint32_t>> work;
    work.push_back({0, 0});

    while (!work.empty()) {
        const auto [node_index, depth] = work.back();
        work.pop_back();

        const std::uint32_t first = bvh.nodes[node_index].first_or_left;
        const std::uint32_t count = bvh.nodes[node_index].count;

        // Node bounds, and the bounds of the centroids (used for binning)
        Box box = empty_box();
        Box centroid_box = empty_box();

        for (std::uint32_t i = 0; i < count; ++i) {
            const TriangleInfo& t = info[bvh.tri_indices[first + i]];
            grow(box, t.box);
            grow(centroid_box, t.centroid);
        }

        bvh.nodes[node_index].box_min = box.min;
        bvh.nodes[node_index].box_max = box.max;

        if (count <= 1 || depth >= kBvhMaxDepth) {
            continue;  // stays a leaf
        }

        const Split split = find_split(
            info, bvh.tri_indices, first, count, centroid_box);

        // Keep a small node as a leaf when splitting would not pay off.
        // Cost of a leaf: count * area. Cost of a split: the sum over
        // both children (traversal cost taken as one triangle test)
        const float leaf_cost =
            static_cast<float>(count) * half_area(box);
        const float split_cost = half_area(box) + split.cost;

        if (split.axis < 0 ||
            (count <= kMaxLeafSize && split_cost >= leaf_cost)) {
            continue;
        }

        // Partition the range: bins [0, split.bin] to the left
        const float lo = axis_value(centroid_box.min, split.axis);
        const float hi = axis_value(centroid_box.max, split.axis);
        const float scale = static_cast<float>(kBins) / (hi - lo);

        auto* begin = bvh.tri_indices.data() + first;
        auto* end = begin + count;

        auto* middle = std::partition(begin, end, [&](std::uint32_t t) {
            const int b = std::min(
                kBins - 1,
                static_cast<int>(
                    (axis_value(info[t].centroid, split.axis) - lo) *
                    scale));
            return b <= split.bin;
        });

        const std::uint32_t left_count =
            static_cast<std::uint32_t>(middle - begin);

        if (left_count == 0 || left_count == count) {
            continue;  // cannot happen: find_split only accepts real splits
        }

        // Children are stored next to each other
        const std::uint32_t left =
            static_cast<std::uint32_t>(bvh.nodes.size());

        bvh.nodes.push_back({{}, first, {}, left_count});
        bvh.nodes.push_back({{}, first + left_count, {},
                             count - left_count});

        bvh.nodes[node_index].first_or_left = left;
        bvh.nodes[node_index].count = 0;

        work.push_back({left, depth + 1});
        work.push_back({left + 1, depth + 1});
    }

    return bvh;
}

BvhStats bvh_stats(const Bvh& bvh) {
    BvhStats stats;
    stats.node_count = static_cast<std::uint32_t>(bvh.nodes.size());

    if (bvh.nodes.empty()) {
        return stats;
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> work;
    work.push_back({0, 0});

    while (!work.empty()) {
        const auto [index, depth] = work.back();
        work.pop_back();

        const BvhNode& node = bvh.nodes[index];
        stats.max_depth = std::max(stats.max_depth, depth);

        if (node.count > 0) {
            ++stats.leaf_count;
            stats.max_leaf_size = std::max(stats.max_leaf_size, node.count);
        } else {
            work.push_back({node.first_or_left, depth + 1});
            work.push_back({node.first_or_left + 1, depth + 1});
        }
    }

    return stats;
}

} // namespace gfx
