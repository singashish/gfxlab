#pragma once

// The path tracer ("Rendered" mode): materials, light sampling,
// trace_path and its kernels.
//
// This file is included ONLY by cuda_raster.cu, inside its anonymous
// namespace and in the same translation unit, at the place where its text
// used to be (the order of the definitions matters). It is a header and
// not a second .cu on purpose: a second .cu would need relocatable device
// code and device linking, which this project does not use. Everything
// the code refers to is declared above the #include in cuda_raster.cu.

// ---------------------------------------------------------------------
// Path tracer ("Rendered" mode). The maths and the ideas (linear light,
// Monte Carlo, cosine sampling, next-event estimation, Russian roulette)
// are explained in core/path_math.h; the constants (bounces, sun, sky)
// are defined there. Surfaces are a metal/dielectric mix: a Lambert
// diffuse layer and a GGX specular lobe (the maths and the reasons are in
// core/brdf.h), plus emission.
//
// Colours: the stored colours (material colour times texel, the values
// the viewport shows) are display (sRGB) values. The path tracer decodes
// them to linear light for the albedo, one rule for textures and
// factors alike, and encodes only the final picture.
//
// The sky and the sun are lights, not scenery: bounce rays that miss
// collect the sky, surfaces get the sun through a shadow ray, but a ray
// that leaves the camera and misses everything sees neither. It gets the
// viewport background (PtParams::background), which goes through the same
// accumulation and tone mapping as any radiance, so the edge pixels of an
// object blend with the background correctly.
// ---------------------------------------------------------------------

// The material of a path-traced surface point, in the units the path
// tracer computes in (linear light).
struct PathMaterial {
    Vec3 albedo;      // linear, 0..1: the fraction of light reflected
    float roughness;  // 0..1: 0 = mirror-like, 1 = fully rough
    float metallic;   // 0..1: 0 = dielectric (plastic, ...), 1 = metal

    // Reflectance of a dielectric looking straight at it (4% for the
    // usual materials, kDielectricF0). 0 means "no glossy layer at all":
    // together with metallic 0 the surface is then plain Lambert paint
    // (that is what turning the materials off gives)
    float dielectric_f0;

    // Linear radiance the surface gives off by itself, per colour
    // channel, in the same units as the sky and the sun. Radiance is
    // "how much light travels along one ray", so this is exactly what
    // a ray that arrives at the surface from the camera side brings
    // back: the surface shines this much, seen from any direction and
    // from both sides. May be far above 1 (a lamp)
    Vec3 emission;
};

// The light a material gives off at the texture coordinates uv: the
// emission factor times the emissive texture (a colour, stored as sRGB,
// so decoded to linear light first). Level 0 only and the same v flip as
// every other texture lookup of the path tracer. Most surfaces give off
// no light: those skip the texture read and the decoding
__device__ __forceinline__ Vec3 surface_emission(
                const MaterialTables& tables,
                const CudaMaterial& material,
                Vec2 uv) {

    if (material.emission.x > 0.0f || material.emission.y > 0.0f ||
        material.emission.z > 0.0f) {
        const Vec3 texel = decode_srgb(
            sample_bilinear(tables.textures[material.emissive_texture],
                            0, uv.x, 1.0f - uv.y));

        return modulate(material.emission, texel);
    }

    return Vec3{0.0f, 0.0f, 0.0f};
}

// Everything the path tracer needs to know about the surface at `point`
// (a hit found by closest_surface). The expensive part of the lookup,
// the extra texture reads, happens here and only here, so the ray tracer
// that shares closest_surface does not pay for it.
//
// Three textures play three different roles, and that decides how their
// texels are read:
//   - base colour: a COLOUR, stored as sRGB. closest_surface already
//     multiplied it into point.base; the albedo is decode_srgb of that.
//   - emissive: also a colour in sRGB, so its texel is decoded to
//     linear light before it scales the (linear) emission factor.
//   - metallic-roughness: DATA, not a colour. The texel values are the
//     numbers themselves (G = roughness, B = metallic, both 0..1 as
//     stored), so decoding them as sRGB would bend the values: a
//     roughness of 0.5 stored as 128 must come out as 0.5, not 0.21.
//
// With materials off (the M key) the surface is plain grey paint: the
// grey albedo that closest_surface put in point.base, no glossy layer
// (dielectric_f0 0, not metallic) and no emission. "No materials" has to
// mean no lamps and no shine either
__device__ __forceinline__ PathMaterial resolve_surface_material(
                const MaterialTables& tables,
                const SurfacePoint& point) {

    PathMaterial result;
    result.albedo = decode_srgb(point.base);

    if (!tables.show_materials) {
        result.roughness = 0.5f;
        result.metallic = 0.0f;
        result.dielectric_f0 = 0.0f;
        result.emission = Vec3{0.0f, 0.0f, 0.0f};
        return result;
    }

    const CudaMaterial& material = tables.materials[point.material];

    // Same texture coordinates and the same v flip as the base colour:
    // image rows run top to bottom while v points up. Level 0 only,
    // like every ray (no pixel footprint to choose a mip level from)
    const float u = point.uv.x;
    const float v = 1.0f - point.uv.y;

    // Data texture: used as stored. The product of two numbers in 0..1
    // stays in 0..1; clamp01 only guards against a table entry that was
    // not made by CudaMeshStore
    const Vec3 packed =
        sample_bilinear(tables.textures[material.metallic_roughness_texture],
                        0, u, v);

    result.roughness = clamp01(material.roughness * packed.y);
    result.metallic = clamp01(material.metallic * packed.z);
    result.dielectric_f0 = kDielectricF0;

    result.emission = surface_emission(tables, material, point.uv);

    return result;
}

// ---------------------------------------------------------------------
// Area lights: sampling the emissive triangles (see core/light_sampling.h)
// ---------------------------------------------------------------------

// A point chosen on a lamp triangle
struct LightSample {
    Vec3 position;
    Vec3 normal;     // unit, orientation arbitrary (lamps are two-sided)
    float area;      // world-space area of the triangle
    float prob;      // probability of having picked this triangle
    Vec3 emission;   // linear radiance leaving the sampled point
};

// The world-space corners of a triangle of an instance
__device__ __forceinline__ void instance_triangle(
                const RtInstance& inst,
                std::uint32_t triangle,
                std::uint32_t& i0,
                std::uint32_t& i1,
                std::uint32_t& i2,
                Vec3& w0,
                Vec3& w1,
                Vec3& w2) {

    const DeviceMesh& mesh = inst.mesh;

    i0 = mesh.indices[triangle * 3 + 0];
    i1 = mesh.indices[triangle * 3 + 1];
    i2 = mesh.indices[triangle * 3 + 2];

    w0 = transform_point(inst.object_to_world, mesh.positions[i0]);
    w1 = transform_point(inst.object_to_world, mesh.positions[i1]);
    w2 = transform_point(inst.object_to_world, mesh.positions[i2]);
}

// Picks a lamp point: the instance by a scan over the instance weights
// (u0), the triangle by binary search in the mesh's cdf (u1), the point
// uniformly by area inside it (u2, u3). Four independent random numbers
// keep the three choices from being correlated. False when the scene has
// no usable lamp
__device__ __forceinline__ bool sample_light(
                const RtScene& scene,
                float u0,
                float u1,
                float u2,
                float u3,
                std::uint32_t& instance,
                std::uint32_t& triangle,
                LightSample& out) {

    const float target = u0 * scene.light_total;
    float running = 0.0f;
    int chosen = -1;
    int last = -1;

    for (int i = 0; i < scene.count; ++i) {
        const float w = scene.items[i].light_weight;

        if (w > 0.0f) {
            last = i;
            running += w;

            if (running >= target) {
                chosen = i;
                break;
            }
        }
    }

    if (chosen < 0) {
        chosen = last;  // rounding: the scan ended just short of target
    }

    if (chosen < 0) {
        return false;
    }

    const RtInstance& inst = scene.items[chosen];
    const DeviceMesh& mesh = inst.mesh;

    float tri_prob = 0.0f;
    const unsigned int pick =
        pick_from_cdf(mesh.light_cdf, mesh.light_count, u1, tri_prob);

    if (!(tri_prob > 0.0f)) {
        return false;
    }

    instance = static_cast<std::uint32_t>(chosen);
    triangle = mesh.light_tris[pick];

    std::uint32_t i0;
    std::uint32_t i1;
    std::uint32_t i2;
    Vec3 p0;
    Vec3 p1;
    Vec3 p2;

    instance_triangle(inst, triangle, i0, i1, i2, p0, p1, p2);
    triangle_area_normal(p0, p1, p2, out.area, out.normal);

    if (!(out.area > 0.0f)) {
        return false;
    }

    float b0;
    float b1;
    float b2;

    sample_triangle_barycentric(u2, u3, b0, b1, b2);

    out.position = p0 * b0 + p1 * b1 + p2 * b2;
    out.prob = (inst.light_weight / scene.light_total) * tri_prob;

    // The emission at the sampled point: factor times emissive texture
    const CudaMaterial& material = scene.surfaces.materials[
        mesh.material_base + mesh.triangle_materials[triangle]];

    out.emission = surface_emission(
        scene.surfaces, material, hit_uv(mesh, i0, i1, i2, b0, b1, b2));

    return true;
}

// The solid-angle density with which sample_light would have produced
// the point where a ray from the surface hit the lamp triangle in
// `point`: exactly the same formula, evaluated from the hit. Returns 0
// when the hit is not a lamp triangle of the lists, or it is a case
// the light sampling skips (grazing, or at the surface); the caller then
// gives the bounce ray the full weight 1
__device__ __forceinline__ float light_hit_pdf(
                const RtScene& scene,
                const SurfacePoint& point,
                Vec3 unit_dir) {

    const RtInstance& inst = scene.items[point.instance];
    const DeviceMesh& mesh = inst.mesh;

    const unsigned int index = find_in_sorted(
        mesh.light_tris, mesh.light_count, point.triangle);

    if (index >= mesh.light_count || !(inst.light_weight > 0.0f)) {
        return 0.0f;
    }

    std::uint32_t i0;
    std::uint32_t i1;
    std::uint32_t i2;
    Vec3 p0;
    Vec3 p1;
    Vec3 p2;
    float area;
    Vec3 normal;

    instance_triangle(inst, point.triangle, i0, i1, i2, p0, p1, p2);
    triangle_area_normal(p0, p1, p2, area, normal);

    const float cos_light = fabsf(dot(normal, unit_dir));

    if (!(area > 0.0f) || cos_light < kMinLightCosine ||
        point.t < kMinLightDistance) {
        return 0.0f;
    }

    const float prob =
        (inst.light_weight / scene.light_total) *
        cdf_probability(mesh.light_cdf, mesh.light_count, index);

    return area_to_solid_angle_pdf(prob / area, point.t, cos_light);
}

// Kd / pi + f_s(l, v): the whole BRDF of the surface for the light
// direction l, and the cosine of h = normalize(l + v) with the normal
// (the density of the bounce needs it too)
__device__ __forceinline__ Vec3 surface_brdf(
                Vec3 n,
                Vec3 v,
                Vec3 l,
                float cos_v,
                float cos_l,
                float alpha,
                Vec3 kd,
                Vec3 f0,
                bool has_specular,
                float& n_dot_h) {

    const Vec3 h = normalize(l + v);

    n_dot_h = dot(n, h);

    Vec3 f = kd * (1.0f / kPi);

    if (has_specular) {
        f = f + ggx_specular_brdf(
            alpha, f0, cos_l, cos_v, n_dot_h, dot(v, h));
    }

    return f;
}

// Adds one contribution (throughput already included) to a sample's
// radiance. An indirect contribution (after a diffuse bounce, or from the
// third ray or vertex on, whatever the lobes were) is first limited to
// `limit`, which removes the "fireflies" of caustic paths (see
// kPtIndirectClamp in core/path_math.h); a direct one is added as it is
__device__ __forceinline__ void add_radiance(
                Vec3& radiance,
                Vec3 contribution,
                bool indirect,
                float limit) {

    radiance = radiance + (indirect
        ? clamp_contribution(contribution, limit)
        : contribution);
}

// The radiance (linear light) arriving at the camera through the point
// (x + jx, y + jy) of the image, for one sample, found by following one
// random path of up to max_bounces surfaces.
//
// For one sample the estimate is
//     L = sum over the path's vertices of (throughput there) * (light
//         added there),
// where the throughput is the product of the weights of the bounces so
// far (for a diffuse bounce the diffuse colour, for a glossy one the
// Fresnel and shadowing factors; see below) and the light added is what
// the vertex's surface emits, the sun it receives through the shadow ray,
// or the sky a ray that left the scene sees.
// Averaging many samples (the accumulation buffer) gives the picture.
//
// Random numbers: every sample of every pixel gets its own seed,
// pcg_hash(pixel ^ pcg_hash(sample_index * 0x9E3779B9u + 1u)), and
// random01 advances it. sample_index is the number of the sample at that
// pixel (the frames since the last restart), so each frame is a new
// independent sample. The multiplication by the golden-ratio constant
// spreads consecutive sample numbers over the 32-bit range before
// hashing; the +1 keeps sample 0 from hashing the number 0.
//
// depth receives the view depth of the camera ray's hit, or -1 for a miss
__device__ __forceinline__ Vec3 trace_path(
                const RtScene& scene,
                const PtParams& pt,
                unsigned int pixel,
                int x,
                int y,
                int width,
                int height,
                unsigned int sample_index,
                float& depth) {

    unsigned int seed =
        pcg_hash(pixel ^ pcg_hash(sample_index * 0x9E3779B9u + 1u));

    // Where inside the pixel the camera ray goes: a uniform point of the
    // pixel's square. Over many samples this averages the pixel's whole
    // footprint, which is free anti-aliasing
    const float jitter_x = random01(seed);
    const float jitter_y = random01(seed);

    Vec3 origin = scene.eye;
    Vec3 dir = camera_ray_dir_at(
        scene, static_cast<float>(x) + jitter_x,
        static_cast<float>(y) + jitter_y, width, height);

    // The camera ray is limited by the near and far plane like in the
    // other modes; the rays after it are not
    float t_min = scene.t_min;
    float t_max = scene.t_max;

    Vec3 radiance{0.0f, 0.0f, 0.0f};
    Vec3 throughput{1.0f, 1.0f, 1.0f};

    depth = -1.0f;

    // The sun as seen by multiple importance sampling (MIS; the long
    // explanation is in core/brdf.h). The sun is found in two ways: the
    // shadow ray below aims at it, and a bounce ray may escape through
    // it. Each way gets a weight from the densities of both, and the two
    // weights add up to 1 for every direction in the cone, so the sun is
    // counted exactly once:
    //     shadow ray:  w_sun  = 1 / (1 + (pdf_bsdf / pdf_sun)^2)
    //     bounce ray:  w_bsdf = 1 / (1 + (pdf_sun / pdf_bsdf)^2)
    // pdf_sun = 1 / sun_solid_angle is the density of the shadow ray's
    // direction (uniform in the cone), pdf_bsdf the density with which
    // the surface's bounce would generate that same direction.
    // sun_radiance = E / solid angle is the radiance inside the cone.
    // prev_pdf is the density of the bounce that produced the current
    // ray (0 for the camera ray, which has no bounce behind it)
    const float sun_solid_angle =
        2.0f * kPi * (1.0f - pt.sun_cos_half_angle);
    const bool sun_mis = pt.sun_enabled && sun_solid_angle > 0.0f;
    const float sun_radiance =
        sun_mis ? pt.sun_irradiance / sun_solid_angle : 0.0f;
    float prev_pdf = 0.0f;

    // True once the path has bounced off the DIFFUSE lobe of a surface.
    // From then on every contribution is clamped (kPtIndirectClamp): such
    // a path is indirect light, and its rare very bright samples are the
    // fireflies. A glossy bounce does not set it, so mirror reflections
    // and glints seen directly are never clamped. Contributions added at
    // the third ray or vertex on (bounce >= 2) are clamped too, whatever
    // the lobes were: the sun seen through two reflections cannot be
    // sampled either (see the loop below, `indirect`)
    bool diffuse_seen = false;

    // Area lights (emissive triangles) get the same treatment as the sun:
    // a shadow ray aimed at a random point of a lamp, and the bounce rays
    // that hit a lamp by chance, weighed against each other so that the
    // light is counted once. With no lamps (or materials off, which
    // switches emission off) none of this runs
    const bool lights_on =
        scene.light_total > 0.0f && scene.surfaces.show_materials;

    // One iteration per ray. Rays 0 .. max_bounces - 1 end on surfaces
    // that are shaded; ray max_bounces only collects the sky and the
    // emission of the surface it hits, because it is the continuation of
    // the last shaded surface (and a miss of the camera ray, ray 0,
    // collects the viewport background instead)
    for (int bounce = 0; bounce <= pt.max_bounces; ++bounce) {
        // Is what this ray and this vertex add indirect light?
        const bool indirect = diffuse_seen || bounce >= 2;

        SurfacePoint point;

        if (!closest_surface(scene, origin, dir, t_min, t_max, point)) {
            const Vec3 unit_dir = normalize(dir);
            const Vec3 seen = bounce == 0
                ? pt.background
                : sky_radiance(unit_dir) * pt.sky_scale;

            add_radiance(radiance, modulate(throughput, seen),
                         indirect, pt.indirect_clamp);

            // A bounce ray that escapes through the sun's cone found the
            // sun by chance. It counts, with the weight w_bsdf of the MIS
            // above; the shadow ray of the surface it left counts the
            // same light with w_sun, and the two weights add up to 1.
            // The final ray (bounce == max_bounces) must count too:
            // the shadow ray of the last shaded surface already used a
            // weight w_sun < 1, and the rest of the sun's light is only
            // accounted for by this ray. The camera ray shows no sun disc
            // (the plain background), and a bounce ray that hits
            // geometry is not an escape: the surface blocks the sun
            if (bounce > 0 && sun_mis && prev_pdf > 0.0f &&
                dot(unit_dir, scene.sun_world) >= pt.sun_cos_half_angle) {
                const float ratio = 1.0f / (prev_pdf * sun_solid_angle);

                add_radiance(radiance, modulate(throughput,
                    Vec3{sun_radiance, sun_radiance, sun_radiance}) *
                    (1.0f / (1.0f + ratio * ratio)),
                    indirect, pt.indirect_clamp);
            }

            break;
        }

        if (bounce == 0) {
            depth = point.t;
        }

        // The albedo is the display colour decoded to linear light. Base
        // colours are clamped to 0..1 by the decoding, so a surface never
        // reflects more than it receives
        const PathMaterial material =
            resolve_surface_material(scene.surfaces, point);
        const Vec3 albedo = material.albedo;

        // Emission: this surface is itself a source of light. The ray
        // that found it travels backwards along the path, so whatever
        // the surface emits towards us is light that reaches the camera
        // after being reflected by all the surfaces between (the
        // throughput, the product of their weights, says how much of it
        // survives). The camera ray's first hit has throughput 1, so a
        // lamp seen directly shows its full emission.
        //
        // This comes BEFORE the "out of bounces" test: the last ray of
        // the path may well end on a lamp, and that light is as real as
        // any other. Only the shading below (the sun, the next
        // direction) is limited to max_bounces surfaces; adding what the
        // hit surface emits costs no further ray, so it is free to count
        // every hit. (Dropping it would make small lamps vanish from the
        // image whenever the path that finds them is already long.)
        // A bounce ray that hits a lamp by chance gets the weight w_bsdf
        // = 1 / (1 + (pdf_light / pdf_bsdf)^2) of the MIS: pdf_light is
        // the density with which the light sampling of the previous
        // surface would have picked this very point, pdf_bsdf (prev_pdf)
        // the density of the bounce that found it. Where light sampling
        // is better (small, distant lamps) the weight is near 0 and
        // that strategy's share is used instead (below). The camera ray
        // (bounce 0) sees the lamp directly: weight 1. This applies to
        // the final ray too: the previous surface's shadow ray used a
        // weight below 1, and this ray supplies the rest
        float emission_weight = 1.0f;

        if (bounce > 0 && lights_on && prev_pdf > 0.0f &&
            (material.emission.x > 0.0f || material.emission.y > 0.0f ||
             material.emission.z > 0.0f)) {
            const float pdf_light =
                light_hit_pdf(scene, point, normalize(dir));

            if (pdf_light > 0.0f) {
                const float ratio = pdf_light / prev_pdf;

                emission_weight = 1.0f / (1.0f + ratio * ratio);
            }
        }

        add_radiance(radiance,
            modulate(throughput, material.emission) * emission_weight,
            indirect, pt.indirect_clamp);

        if (bounce == pt.max_bounces) {
            break;  // out of bounces: this surface is not shaded
        }

        Vec3 n = normalize(point.normal);

        if (dot(n, n) == 0.0f) {
            n = point.geometric;  // degenerate vertex normals
        }

        const Vec3 start =
            point.position + point.geometric * shadow_offset(point.position);

        // The surface is a mix of two lobes, a diffuse (Lambert) one and a
        // glossy (GGX) one; core/brdf.h explains the model. Everything
        // below is in world space. v points back to where the ray came
        // from, and its cosine with the normal is kept above a tiny
        // number, because interpolated normals can face slightly away
        // from the viewer
        const Vec3 v = -normalize(dir);
        const float cos_v = fmaxf(dot(n, v), kMinCosView);
        const float alpha = ggx_alpha_from_roughness(material.roughness);

        // No glossy layer at all (materials off): the specular lobe is
        // skipped entirely. A dielectric f0 of 0 is the "no layer" marker
        // of PathMaterial; Schlick's curve through 0 would still rise to 1
        // at glancing angles
        const bool has_specular =
            material.dielectric_f0 > 0.0f || material.metallic > 0.0f;

        // Reflectance at normal incidence: a dielectric reflects 4% of
        // every colour, a metal reflects its colour (the base colour)
        Vec3 f0{0.0f, 0.0f, 0.0f};

        if (has_specular) {
            const float m = material.metallic;
            const float d0 = material.dielectric_f0 * (1.0f - m);

            f0 = Vec3{d0 + albedo.x * m, d0 + albedo.y * m,
                      d0 + albedo.z * m};
        }

        // The diffuse colour: nothing for a metal (its electrons reflect
        // or absorb all light at the surface), and for a dielectric the
        // albedo reduced by what the clear glossy layer on top already
        // reflected. The more it reflects at this view angle (glancing
        // views), the less is left to enter the diffuse layer. A cheap
        // approximation of the energy balance, not a measured model
        const float fd =
            fresnel_dielectric_layer(material.dielectric_f0, cos_v);
        const Vec3 kd = albedo * ((1.0f - material.metallic) * (1.0f - fd));

        // The probability of bouncing off the glossy lobe (see below); it
        // is also needed here, to weigh the shadow ray against the bounce
        const float p_spec = specular_lobe_probability(
            has_specular ? fresnel_schlick(f0, cos_v)
                         : Vec3{0.0f, 0.0f, 0.0f},
            kd);

        // Next-event estimation: one shadow ray towards a random point
        // of the sun's disc. The disc has radiance E / solid_angle and a
        // uniformly chosen direction has density 1 / solid_angle, so the
        // solid angle cancels and the surface receives
        //     (Kd / pi + f_s(l, v)) * E * cos(angle to the normal),
        // both lobes at once (the lobes are not picked here: the sun is
        // a light to aim at, not a direction to bounce into). This
        // is multiplied by the MIS weight w_sun, which is close to 1
        // where the bounce rays would rarely find the sun (diffuse and
        // rough surfaces: pdf_bsdf is about a thousand times smaller than
        // pdf_sun, so the weight is 1 to six digits and nothing changes
        // for them) and close to 0 on a smooth glossy highlight, where
        // the bounce rays do the job better. Directions below the
        // surface (shading normal or the real triangle) get nothing
        if (pt.sun_enabled) {
            const float u1 = random01(seed);
            const float u2 = random01(seed);

            const Vec3 l = sample_sun_cone(
                scene.sun_world, pt.sun_cos_half_angle, u1, u2);

            const float cos_l = dot(n, l);

            if (cos_l > 0.0f && dot(point.geometric, l) > 0.0f &&
                !scene_occluded(scene, start, l, 1.0e30f)) {
                float n_dot_h;
                const Vec3 f = surface_brdf(
                    n, v, l, cos_v, cos_l, alpha, kd, f0, has_specular,
                    n_dot_h);

                // pdf_bsdf / pdf_sun = pdf_bsdf * solid angle
                float w_sun = 1.0f;

                if (sun_mis) {
                    const float ratio = sun_solid_angle *
                        bsdf_mixture_pdf(p_spec, alpha, cos_v, cos_l,
                                         n_dot_h);

                    w_sun = 1.0f / (1.0f + ratio * ratio);
                }

                add_radiance(radiance, modulate(throughput, f) *
                    (pt.sun_irradiance * cos_l * w_sun),
                    indirect, pt.indirect_clamp);
            }
        }

        // Next-event estimation for the area lights: one shadow ray
        // towards a random point of a random lamp triangle. A point on
        // the lamp has the solid-angle density pdf_light (see
        // core/light_sampling.h), so the light the surface receives is
        //     f(l) cos(l) L_e / pdf_light,
        // times the MIS weight w_light = 1 / (1 + (pdf_bsdf / pdf_light)^2)
        // (pdf_bsdf: the density with which this surface's bounce would
        // have generated l). Written as one fraction,
        //     f cos L_e / (pdf_light + pdf_bsdf^2 / pdf_light),
        // so a tiny pdf_light cannot turn "huge times zero" into a NaN.
        // Lamps are two-sided, and the lamp triangle the surface point
        // itself is on is skipped (it cannot light itself)
        if (lights_on) {
            const float lu0 = random01(seed);
            const float lu1 = random01(seed);
            const float lu2 = random01(seed);
            const float lu3 = random01(seed);

            std::uint32_t light_instance = 0u;
            std::uint32_t light_triangle = 0u;
            LightSample ls;

            if (sample_light(scene, lu0, lu1, lu2, lu3,
                             light_instance, light_triangle, ls) &&
                !(light_instance == point.instance &&
                  light_triangle == point.triangle)) {
                const Vec3 to_light = ls.position - start;
                const float dist = length(to_light);

                if (dist >= kMinLightDistance) {
                    const Vec3 l = to_light * (1.0f / dist);
                    const float cos_l = dot(n, l);
                    const float cos_light = fabsf(dot(ls.normal, l));

                    if (cos_l > 0.0f && dot(point.geometric, l) > 0.0f &&
                        cos_light >= kMinLightCosine &&
                        !scene_occluded(
                            scene, start, l,
                            dist * (1.0f - kLightShadowShortening))) {
                        const float pdf_light = area_to_solid_angle_pdf(
                            ls.prob / ls.area, dist, cos_light);

                        float n_dot_h;
                        const Vec3 f = surface_brdf(
                            n, v, l, cos_v, cos_l, alpha, kd, f0,
                            has_specular, n_dot_h);
                        const float pdf_bsdf = bsdf_mixture_pdf(
                            p_spec, alpha, cos_v, cos_l, n_dot_h);

                        if (pdf_light > 0.0f && finite_float(pdf_light)) {
                            const float scale = cos_l /
                                (pdf_light + pdf_bsdf * pdf_bsdf /
                                                 pdf_light);

                            add_radiance(radiance,
                                modulate(throughput,
                                         modulate(f, ls.emission)) * scale,
                                indirect, pt.indirect_clamp);
                        }
                    }
                }
            }
        }

        // Continue the path in ONE direction, sampled from ONE of the two
        // lobes. The two random numbers v1 and v2 are used by whichever
        // lobe is picked. The lobe is picked with probability p_spec for
        // the glossy one and 1 - p_spec for the diffuse one (the
        // probability follows how much light each lobe reflects). A third
        // random number is only drawn when there is really a choice
        // (0 < p_spec < 1), so a surface with a single lobe uses the same
        // random numbers as before the glossy lobe existed
        const float v1 = random01(seed);
        const float v2 = random01(seed);
        const float v3 = (p_spec > 0.0f && p_spec < 1.0f)
            ? random01(seed) : 0.0f;

        const BsdfSample bounce_sample =
            sample_bsdf_mixture(n, v, alpha, p_spec, v1, v2, v3);

        const Vec3 d = bounce_sample.d;
        const float cos_out = dot(n, d);

        // A glossy bounce can end up below the surface (the microfacet's
        // mirror points into the surface), and a diffuse one can dip below
        // the actual triangle (smooth normals bend away from it): such a
        // sample carries no light, the path ends
        if (cos_out <= 0.0f || dot(d, point.geometric) <= 0.0f) {
            break;
        }

        // The density with which this bounce generated d: the mixture of
        // both lobes, whichever one was picked (core/brdf.h). It is the
        // divisor of the weight below and, kept for the next ray, the
        // pdf_bsdf of the MIS weight if that ray escapes through the sun
        const Vec3 h_out = normalize(d + v);
        const float n_dot_h_out = dot(n, h_out);
        const float pdf = bsdf_mixture_pdf(
            p_spec, alpha, cos_v, cos_out, n_dot_h_out);

        if (!finite_float(pdf) || !(pdf > 0.0f)) {
            break;
        }

        // The weight of the bounce, f(d) cos(d) / pdf(d), with the
        // mixture density. Written out like this it is the same for any
        // lobe that gets picked, which is what makes it consistent with
        // the MIS weights. When the surface has a single lobe the density
        // cancels the BRDF almost completely and the plain forms are used
        // (they are exact and well behaved):
        //  - glossy only: F(v.h) G2 / G1(v), at most 1 per channel (the
        //    cancellation is explained in core/brdf.h)
        //  - diffuse only: Kd (the density cos / pi cancels Kd / pi cos)
        // so for them the result is the same as before MIS. With both
        // lobes the full f = Kd / pi + f_s is divided by the mixture
        Vec3 weight;

        if (p_spec >= 1.0f) {
            weight = fresnel_schlick(f0, dot(v, bounce_sample.h)) *
                     smith_g2_over_g1(alpha, cos_out, cos_v);
        } else if (p_spec <= 0.0f) {
            weight = kd;
        } else {
            const Vec3 f = kd * (1.0f / kPi) +
                ggx_specular_brdf(alpha, f0, cos_out, cos_v, n_dot_h_out,
                                  dot(v, h_out));

            weight = f * (cos_out / pdf);
        }

        prev_pdf = pdf;

        if (!bounce_sample.specular) {
            diffuse_seen = true;  // what is added from here on is indirect
        }

        throughput = modulate(throughput, weight);

        // Russian roulette: terminate with probability 1 - q, otherwise
        // divide by q, which keeps the average the same
        if (bounce >= kPtRouletteStart) {
            const float q = russian_roulette_survival(throughput);

            if (random01(seed) >= q) {
                break;
            }

            throughput = throughput * (1.0f / q);
        }

        origin = start;
        dir = d;
        t_min = 0.0f;
        t_max = 1.0e30f;
    }

    return radiance;
}

// Adds pt.samples_to_add samples to every pixel's accumulator. A sum is
// a running mean in disguise: the mean after N samples is sum / N, so
// storing the sum and the count (not the mean) needs no rounding
// compensation, and one more sample is just one more addition
__global__ void path_trace_kernel(
                int width,
                int height,
                RtScene scene,
                PtParams pt) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const unsigned int pixel =
        static_cast<unsigned int>(pixel_index(x, y, width));

    // Starting over: whatever is in the buffer is from an older picture
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, -1.0f);

    if (pt.samples_before > 0) {
        acc = pt.accum[pixel];
    }

    for (unsigned int i = 0; i < pt.samples_to_add; ++i) {
        float depth = -1.0f;

        const Vec3 radiance = trace_path(
            scene, pt, pixel, x, y, width, height,
            pt.samples_before + i, depth);

        // A NaN or infinite sample adds nothing (it still counts as a
        // sample), so one bad value can never poison the sum
        if (finite_vec3(radiance)) {
            acc.x += radiance.x;
            acc.y += radiance.y;
            acc.z += radiance.z;
        }

        // The depth the grid is tested against: the closest camera hit
        // of any sample, so a pixel on an object's silhouette keeps
        // hiding the grid instead of flickering with the jitter
        if (depth > 0.0f && (acc.w < 0.0f || depth < acc.w)) {
            acc.w = depth;
        }
    }

    pt.accum[pixel] = acc;
}

// Always runs: turns the accumulated sums into the picture. Idempotent
// (it only reads the accumulator), so a converged picture is just
// resolved again every frame
__global__ void pt_resolve_kernel(
                uchar4* color,
                unsigned long long* visibility,
                int width,
                int height,
                RtScene scene,
                PtParams pt) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t pixel = pixel_index(x, y, width);

    const unsigned int count = pt.samples_before + pt.samples_to_add;

    if (count == 0) {
        return;  // nothing accumulated yet: keep the cleared background
    }

    const float4 acc = pt.accum[pixel];

    // Every sample so far missed everything: the pixel is background and
    // keeps exactly the colour the clear pass gave it, with no visibility
    // entry, so the grid draws over it. (Their mean would be the same
    // colour up to rounding; this makes it exact.) A pixel with at least
    // one hit is resolved below, its mean including the background
    // radiance of the samples that missed: the anti-aliased edge
    if (acc.w < 0.0f) {
        return;
    }

    const Vec3 mean =
        Vec3{acc.x, acc.y, acc.z} * (1.0f / static_cast<float>(count));

    const Vec3 display = linear_to_display(mean, pt.exposure);

    color[pixel] = make_uchar4(
        to_byte(display.x), to_byte(display.y), to_byte(display.z), 255);

    // Covered pixel: depth + id 1 as the ray tracer writes them. The
    // world position is on the pixel centre's ray at the stored view
    // depth; the depth value only depends on the view depth, so the
    // jitter of the sample that found it does not matter
    if (acc.w > 0.0f) {
        const Vec3 position =
            scene.eye + acc.w * camera_ray_dir(scene, x, y, width, height);

        visibility[pixel] = visibility_entry(scene, position);
    }
}
