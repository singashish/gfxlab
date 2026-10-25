#pragma once

// Ray-tracing device helpers shared by the ray tracer ("T" mode) and
// the path tracer: shadow and ambient-occlusion rays, camera rays,
// closest_surface, SurfacePoint, the per-pixel random numbers and the
// wavefront bookkeeping helpers.
//
// This file is included ONLY by cuda_raster.cu, inside its anonymous
// namespace and in the same translation unit, at the place where its text
// used to be (the order of the definitions matters). It is a header and
// not a second .cu on purpose: a second .cu would need relocatable device
// code and device linking, which this project does not use. Everything
// the code refers to is declared above the #include in cuda_raster.cu.

// Does anything in the scene block the ray origin + t * dir for
// 0 < t < t_max? Used for shadow and AO rays, so any hit is enough.
// dir is unit length in world space, so t_max is a world distance; the
// object-space direction is not renormalized, so t means the same there
__device__ __forceinline__ bool scene_occluded(
                const RtScene& scene,
                Vec3 origin,
                Vec3 dir,
                float t_max) {

    for (int i = 0; i < scene.count; ++i) {
        const RtInstance& inst = scene.items[i];

        const Vec3 local_origin =
            transform_point(inst.world_to_object, origin);
        const Vec3 local_dir =
            transform_direction(inst.world_to_object, dir);

        // Both trees hold the same triangles in the same order, so
        // either answers the same question
        const bool blocked = scene.use_bvh4
            ? bvh4_occluded(inst.mesh.bvh4_nodes,
                            inst.mesh.bvh_tri_indices,
                            inst.mesh.positions,
                            inst.mesh.indices,
                            local_origin, local_dir, 0.0f, t_max)
            : bvh_occluded(inst.mesh.bvh_nodes,
                           inst.mesh.bvh_tri_indices,
                           inst.mesh.positions,
                           inst.mesh.indices,
                           local_origin, local_dir, 0.0f, t_max);

        if (blocked) {
            return true;
        }
    }

    return false;
}

// Distance to move a shadow ray's start off the surface, so it does not
// hit the triangle it starts on ("shadow acne"). Scaled with the size of
// the coordinates because float precision is relative
__device__ __forceinline__ float shadow_offset(Vec3 p) {
    const float largest =
        fmaxf(fabsf(p.x), fmaxf(fabsf(p.y), fabsf(p.z)));

    return 1.0e-4f * fmaxf(1.0f, largest);
}

// PCG hash: turns any 32-bit number into a well-scrambled one. Chained,
// it gives each pixel and sample its own repeatable random sequence
__device__ __forceinline__ unsigned int pcg_hash(unsigned int v) {
    const unsigned int state = v * 747796405u + 2891336453u;
    const unsigned int word =
        ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Next random number in [0, 1) from the seed, advancing the seed
__device__ __forceinline__ float random01(unsigned int& seed) {
    seed = pcg_hash(seed);
    return static_cast<float>(seed >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------
// Ray tracer building blocks, shared by both ways of running it:
//  - the "megakernel": one thread does everything for one pixel
//  - the "wavefront": the work is split into passes, and each shadow or
//    AO ray gets its own thread (see cuda_ray_trace below)
// ---------------------------------------------------------------------

// Camera ray direction through the point (px, py) of the image, in
// pixels: (0, 0) is the top-left corner of the first pixel and
// (x + 0.5, y + 0.5) the centre of pixel (x, y). Not unit length: its
// view-space z is exactly -1, so t is the view depth (for any px, py)
__device__ __forceinline__ Vec3 camera_ray_dir_at(
                const RtScene& scene,
                float px,
                float py,
                int width,
                int height) {

    const float ndc_x = 2.0f * px / width - 1.0f;
    const float ndc_y = 1.0f - 2.0f * py / height;

    return scene.forward + ndc_x * scene.right + ndc_y * scene.up;
}

// The ray through the centre of pixel (x, y)
__device__ __forceinline__ Vec3 camera_ray_dir(
                const RtScene& scene,
                int x,
                int y,
                int width,
                int height) {

    return camera_ray_dir_at(scene, x + 0.5f, y + 0.5f, width, height);
}

// The point a ray hit, with everything shading needs, in world space.
// Both normals are turned towards the side the camera sees
struct SurfacePoint {
    float t;         // distance along the ray (view depth for camera rays)
    Vec3 position;
    Vec3 normal;     // interpolated shading normal (not unit length)
    Vec3 geometric;  // the triangle's own normal (unit length)
    Vec3 base;       // surface colour: material colour times texel

    // What the path tracer needs to look up the rest of the material
    // (see resolve_surface_material). Both are only filled while
    // materials are shown; with them off nothing reads them. They are
    // cheap facts the hit has found out anyway, so recording them costs
    // the ray tracer nothing; the expensive part (more texture reads) is
    // left to the path tracer
    std::uint32_t material;  // index into the renderer-wide table
    Vec2 uv;                 // texture coordinates at the hit (v up)

    // Which instance (index into RtScene::items) and which triangle of
    // its mesh was hit. The path tracer needs them to work out how likely
    // the light sampling would have been to pick this very point of a
    // lamp (multiple importance sampling). Two plain stores
    std::uint32_t instance;
    std::uint32_t triangle;
};

// The texture coordinates at a hit on the triangle with corners
// (i0, i1, i2) and barycentric weights (w0, w1, w2). A mesh without
// texture coordinates uses uv (0, 0)
__device__ __forceinline__ Vec2 hit_uv(
                const DeviceMesh& mesh,
                std::uint32_t i0,
                std::uint32_t i1,
                std::uint32_t i2,
                float w0,
                float w1,
                float w2) {

    Vec2 uv{0.0f, 0.0f};

    if (mesh.texcoords) {
        uv = mesh.texcoords[i0] * w0 +
             mesh.texcoords[i1] * w1 +
             mesh.texcoords[i2] * w2;
    }

    return uv;
}

// The surface colour at a ray hit: the material's colour times its
// base-colour texture, at the texture coordinates uv. (The caller uses
// the plain grey instead when materials are off.)
//
// Camera rays have no pixel footprint to measure, so the texture is
// sampled at level 0 only (bilinear, no mip levels). The price is some
// shimmer on finely detailed textures far away or at grazing angles,
// which the rasterizer avoids with trilinear filtering. Mip levels for
// rays would need ray differentials, which is a step for later
__device__ __forceinline__ Vec3 hit_base_color(
                const MaterialTables& tables,
                const CudaMaterial& material,
                Vec2 uv) {

    // Image rows run top to bottom, v points up: sample at 1 - v
    const Vec3 texel = sample_bilinear(
        tables.textures[material.texture], 0, uv.x, 1.0f - uv.y);

    return Vec3{material.base_color.x * texel.x,
                material.base_color.y * texel.y,
                material.base_color.z * texel.z};
}

// Closest hit of the ray origin + t * dir with t_min < t < t_max over
// every instance. Camera rays start at the eye and use the near and far
// plane as their range; the rays of a path tracer start at an offset
// surface point (see shadow_offset), with t_min 0 and a huge t_max.
// The direction is moved into object space without normalizing it, so t
// means the same distance in both spaces and hits can be compared
// directly. Fills point on a hit
__device__ __forceinline__ bool closest_surface(
                const RtScene& scene,
                Vec3 origin,
                Vec3 dir,
                float t_min,
                float t_max,
                SurfacePoint& point) {

    float best_t = t_max;
    int best_item = -1;
    BvhHit best{};

    for (int i = 0; i < scene.count; ++i) {
        const RtInstance& inst = scene.items[i];

        BvhHit hit;

        const Vec3 local_origin =
            transform_point(inst.world_to_object, origin);
        const Vec3 local_dir =
            transform_direction(inst.world_to_object, dir);

        const bool found = scene.use_bvh4
            ? bvh4_trace(inst.mesh.bvh4_nodes,
                         inst.mesh.bvh_tri_indices,
                         inst.mesh.positions,
                         inst.mesh.indices,
                         local_origin, local_dir,
                         t_min, best_t, hit)
            : bvh_trace(inst.mesh.bvh_nodes,
                        inst.mesh.bvh_tri_indices,
                        inst.mesh.positions,
                        inst.mesh.indices,
                        local_origin, local_dir,
                        t_min, best_t, hit);

        if (found) {
            best_t = hit.t;
            best = hit;
            best_item = i;
        }
    }

    if (best_item < 0) {
        return false;
    }

    const RtInstance& inst = scene.items[best_item];
    const DeviceMesh& mesh = inst.mesh;

    const std::uint32_t i0 = mesh.indices[best.triangle * 3 + 0];
    const std::uint32_t i1 = mesh.indices[best.triangle * 3 + 1];
    const std::uint32_t i2 = mesh.indices[best.triangle * 3 + 2];

    const Vec3 p0 = mesh.positions[i0];
    const Vec3 p1 = mesh.positions[i1];
    const Vec3 p2 = mesh.positions[i2];

    // Geometric normal in object space, from the triangle's winding
    const Vec3 geometric = cross(p1 - p0, p2 - p0);

    // Interpolated vertex normal (barycentrics u, v from the hit)
    const float w0 = 1.0f - best.u - best.v;

    Vec3 n_object = geometric;

    if (mesh.normals) {
        n_object = mesh.normals[i0] * w0 +
                   mesh.normals[i1] * best.u +
                   mesh.normals[i2] * best.v;
    }

    // Front-facing when the ray arrives against the counter-clockwise
    // normal. The sign of this dot product is the same in object and
    // world space, so the object-space direction can be used
    const Vec3 local_dir =
        transform_direction(inst.world_to_object, dir);
    const bool front_facing = dot(geometric, local_dir) < 0.0f;

    point.t = best_t;
    point.instance = static_cast<std::uint32_t>(best_item);
    point.triangle = best.triangle;
    point.position = origin + best_t * dir;
    point.normal = transform_direction(inst.normal_to_world, n_object);
    point.geometric = normalize(
        transform_direction(inst.normal_to_world, geometric));

    if (!front_facing) {
        point.normal = -point.normal;
        point.geometric = -point.geometric;
    }

    if (scene.surfaces.show_materials) {
        point.material =
            mesh.material_base + mesh.triangle_materials[best.triangle];
        point.uv = hit_uv(mesh, i0, i1, i2, w0, best.u, best.v);
        point.base = hit_base_color(
            scene.surfaces, scene.surfaces.materials[point.material],
            point.uv);
    } else {
        point.material = 0;
        point.uv = Vec2{0.0f, 0.0f};
        point.base = unlit_base_color();
    }

    return true;
}

// Does the sun reach this point? Returns true when a shadow ray was
// needed (the surface faces the sun) and sets blocked
__device__ __forceinline__ bool trace_shadow(
                const RtScene& scene,
                const SurfacePoint& point,
                bool& blocked) {

    blocked = false;

    if (!scene.shadows ||
        dot(normalize(point.normal), scene.sun_world) <= 0.0f) {
        return false;  // surfaces facing away are unlit by the sun anyway
    }

    const Vec3 origin =
        point.position + point.geometric * shadow_offset(point.position);

    blocked = scene_occluded(scene, origin, scene.sun_world, 1.0e30f);
    return true;
}

// One ambient occlusion ray for the given pixel and sample number.
// Returns true when a ray was traced and sets escaped when it got away
// without hitting anything within the AO radius
__device__ __forceinline__ bool trace_ao_ray(
                const RtScene& scene,
                const SurfacePoint& point,
                unsigned int pixel,
                unsigned int sample,
                bool& escaped) {

    escaped = false;

    unsigned int seed = pcg_hash(pixel ^ pcg_hash(sample));

    const float u1 = random01(seed);
    const float u2 = random01(seed);

    const Vec3 d = cosine_hemisphere(normalize(point.normal), u1, u2);

    // A direction that dips below the actual triangle (possible where
    // smooth normals bend away from it) counts as blocked
    if (dot(d, point.geometric) <= 0.0f) {
        return false;
    }

    const Vec3 origin =
        point.position + point.geometric * shadow_offset(point.position);

    escaped = !scene_occluded(scene, origin, d, scene.ao_radius);
    return true;
}

// The visibility buffer entry (depth in the high 32 bits, id 1 in the
// low ones) for a ray-traced surface point. Same depth the rasterizer
// would have written, so the grid's depth test treats all paths alike
__device__ __forceinline__ unsigned long long visibility_entry(
                const RtScene& scene,
                Vec3 position) {

    const Vec4 clip =
        scene.view_proj * make_vec4(position, 1.0f);

    float depth01 = (clip.z / clip.w) * 0.5f + 0.5f;
    depth01 = fminf(fmaxf(depth01, 0.0f), 1.0f);

    return (static_cast<unsigned long long>(__float_as_uint(depth01))
                << 32) |
           1ULL;
}

// Combines the shadow and AO results into the pixel's colour and depth.
// new_escaped is how many of this frame's AO rays got away
__device__ __forceinline__ void finish_pixel(
                uchar4* color,
                unsigned long long* visibility,
                unsigned int pixel,
                const RtScene& scene,
                const SurfacePoint& point,
                bool sun_blocked,
                unsigned int new_escaped) {

    float ao = 1.0f;

    if (scene.ao) {
        float escaped = scene.ao_samples_before > 0
            ? scene.ao_accum[pixel]
            : 0.0f;

        escaped += static_cast<float>(new_escaped);

        const unsigned int total =
            scene.ao_samples_before +
            static_cast<unsigned int>(scene.ao_rays);

        if (scene.ao_rays > 0) {
            scene.ao_accum[pixel] = escaped;
        }

        ao = total > 0 ? escaped / static_cast<float>(total) : 1.0f;
    }

    const Vec3 n_view = transform_direction(scene.view, point.normal);
    const Vec3 view_pos = transform_point(scene.view, point.position);

    color[pixel] = lit_color(
        n_view, view_pos, scene.up_view, scene.sun_view,
        sun_blocked ? 0.0f : 1.0f, ao, point.base);

    visibility[pixel] = visibility_entry(scene, point.position);
}

// Adds a thread's ray counts to the frame's counters: first per block
// in shared memory, then one global atomic per counter per block.
// Every thread of the block must call it (it contains barriers)
__device__ __forceinline__ void add_ray_counts(
                unsigned long long* ray_counts,
                unsigned int primary,
                unsigned int shadow,
                unsigned int ao) {

    __shared__ unsigned int block_counts[3];

    const bool first_thread = threadIdx.x == 0 && threadIdx.y == 0;

    if (first_thread) {
        block_counts[0] = 0;
        block_counts[1] = 0;
        block_counts[2] = 0;
    }

    __syncthreads();

    if (ray_counts != nullptr) {
        if (primary) atomicAdd(&block_counts[0], primary);
        if (shadow)  atomicAdd(&block_counts[1], shadow);
        if (ao)      atomicAdd(&block_counts[2], ao);
    }

    __syncthreads();

    if (first_thread && ray_counts != nullptr) {
        atomicAdd(&ray_counts[0],
                  static_cast<unsigned long long>(block_counts[0]));
        atomicAdd(&ray_counts[1],
                  static_cast<unsigned long long>(block_counts[1]));
        atomicAdd(&ray_counts[2],
                  static_cast<unsigned long long>(block_counts[2]));
    }
}
