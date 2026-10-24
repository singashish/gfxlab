#pragma once

/*
 * The glossy surface model of the CUDA path tracer: a GGX specular lobe
 * on top of a Lambert diffuse layer (the "metallic-roughness" model of
 * glTF). Header-only and GFX_HD like path_math.h, so the CPU tests run
 * the very same code as the GPU; nothing here calls a host-only std::
 * function (only sqrtf, sinf, cosf, fminf, fmaxf).
 *
 * All directions point AWAY from the surface point: v towards the viewer
 * (the reverse of the ray that arrived), l towards the light or towards
 * where the next ray goes, n is the surface normal, and h, the "half
 * vector", is the direction halfway between l and v.
 *
 * Microfacets. A rough surface looks flat to the eye, but under a
 * microscope it is a field of tiny mirrors (microfacets), each tilted a
 * little away from the average normal n. A mirror only sends light from l
 * to v when its own normal is exactly h, so the brightness of the glossy
 * highlight is "how many microfacets have the normal h" times "how much
 * of each one is actually visible and lit" times "how much light a
 * mirror of this material reflects". Those three questions are the
 * three factors of the specular BRDF
 *     f_s(l, v) = D(h) * Vis(l, v) * F(v.h)
 * (a BRDF says how much of the light arriving from l leaves towards v):
 *
 *  - D, the normal distribution (ggx_d): what fraction of the surface
 *    has microfacet normal h. GGX (also called Trowbridge-Reitz) has a
 *    sharp peak at h = n and long tails, which is why real highlights
 *    have a bright core with a soft glow around it.
 *  - G, the shadowing-masking term (smith_g1, smith_g2_height_correlated):
 *    at glancing angles microfacets hide behind their neighbours, so
 *    some are not lit (shadowing) or not seen (masking). G1 is the
 *    visible fraction for one direction, G2 for both l and v together.
 *    Vis = G2 / (4 (n.l) (n.v)) is G2 with the BRDF's own denominator
 *    folded in (ggx_visibility).
 *  - F, the Fresnel term (fresnel_schlick): every surface becomes more
 *    mirror-like at glancing angles. A pane of glass reflects 4% of the
 *    light looking straight at it, and nearly everything when looking
 *    along its surface. Schlick's formula is a cheap curve through those
 *    two ends: F0 + (1 - F0) (1 - cos)^5, where F0 is the reflectance
 *    at normal incidence. It is an approximation of the real Fresnel
 *    equations, accurate to a few percent for common materials.
 *
 * Roughness and alpha. Artists set "roughness" 0..1. The distribution
 * uses alpha = roughness^2: the squaring spends more of the slider on
 * the nearly smooth end, where the eye is most sensitive to changes
 * (alpha grows slowly at first, so 0.25 looks half as rough as 0.5
 * rather than almost as rough). A perfect mirror (alpha = 0) is a delta
 * function that no random direction can ever hit, so alpha has a floor
 * (kGgxMinAlpha): a very sharp but finite lobe.
 *
 * Single scattering. This model follows each ray through ONE bounce on
 * the microfacet field. On a rough surface, light also bounces from one
 * microfacet into another before leaving, and GGX ignores that light,
 * so rough metals come out too dark (at roughness 1 by a large part).
 * The fix is a multiple-scattering compensation; it is not part of this
 * step.
 */

#include "core/math_util.h"
#include "core/path_math.h"
#include "core/vec3.h"

#include <cmath>

namespace gfx {

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

// Reflectance of a dielectric (plastic, glass, wood, ...) looking straight
// at it: 4% (index of refraction 1.5)
constexpr float kDielectricF0 = 0.04f;

// Lower limit of alpha (see the note on mirrors above)
constexpr float kGgxMinAlpha = 0.002f;

// The cosine between the normal and the view direction is limited from
// below by this: interpolated vertex normals can face very slightly away
// from the viewer, and the formulas divide by it
constexpr float kMinCosView = 1.0e-4f;

// The path tracer picks the specular lobe for the next bounce with a
// probability between these two (unless one lobe is absent): never
// ignoring a lobe entirely keeps the variance of the estimate bounded
constexpr float kSpecularProbabilityMin = 0.1f;
constexpr float kSpecularProbabilityMax = 0.9f;

// Cosines below this are treated as this (division guard inside the
// Smith functions; never reached by the renderer, which rejects
// directions at or below the horizon first)
constexpr float kTinyCosine = 1.0e-6f;

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

// (1 - x)^5 by multiplication (powf is slower and no more accurate), for
// x clamped to 0..1
GFX_HD inline float one_minus_pow5(float x) {
    const float m = 1.0f - clamp01(x);
    const float m2 = m * m;

    return m2 * m2 * m;
}

GFX_HD inline float max_component(Vec3 v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

// ---------------------------------------------------------------------
// D, G, F
// ---------------------------------------------------------------------

// alpha = roughness^2, kept in kGgxMinAlpha..1 (also for a NaN input)
GFX_HD inline float ggx_alpha_from_roughness(float roughness) {
    return fminf(fmaxf(roughness * roughness, kGgxMinAlpha), 1.0f);
}

// GGX / Trowbridge-Reitz normal distribution:
//     D(h) = alpha^2 / (pi ((n.h)^2 (alpha^2 - 1) + 1)^2)
// normalised so that the integral of D(h) (n.h) over the hemisphere of
// directions h is 1: the projected area of the microfacets adds up to the
// area of the surface. Its unit is 1 / steradian. n_dot_h <= 0 (a
// microfacet turned away from the surface) gives 0
GFX_HD inline float ggx_d(float alpha, float n_dot_h) {
    if (!(n_dot_h > 0.0f)) {
        return 0.0f;
    }

    // A unit vector's cosine can come out a hair above 1 by rounding
    const float c = fminf(n_dot_h, 1.0f);
    const float a2 = alpha * alpha;
    const float c2 = c * c;

    // (n.h)^2 (a2 - 1) + 1 rewritten as sin^2 + cos^2 a2: no cancellation
    // of two nearly equal numbers close to the peak, where the lobe of a
    // smooth surface is only a few thousandths of a radian wide
    const float sin2 = (1.0f - c) * (1.0f + c);
    const float d = sin2 + c2 * a2;

    return a2 / (kPi * d * d);
}

// Smith's Lambda for a direction w with n.w = cos_w > 0: how strongly the
// microfacets hide that direction.
//     Lambda = (sqrt(1 + alpha^2 tan^2) - 1) / 2
//            = (sqrt(alpha^2 + (1 - alpha^2) cos^2) / cos - 1) / 2
// (0 for looking straight down, growing without limit towards the horizon)
GFX_HD inline float smith_lambda(float alpha, float cos_w) {
    const float c = fmaxf(cos_w, kTinyCosine);
    const float a2 = alpha * alpha;

    return 0.5f * (sqrtf(a2 + (1.0f - a2) * c * c) / c - 1.0f);
}

// Fraction of the microfacets visible from one direction: 1 / (1 + Lambda)
// written as 2 c / (c + sqrt(alpha^2 + (1 - alpha^2) c^2)). 0 for a
// direction at or below the horizon, 1 straight on
GFX_HD inline float smith_g1(float alpha, float cos_w) {
    if (!(cos_w > 0.0f)) {
        return 0.0f;
    }

    const float c = fminf(cos_w, 1.0f);
    const float a2 = alpha * alpha;

    return 2.0f * c / (c + sqrtf(a2 + (1.0f - a2) * c * c));
}

// sqrt(alpha^2 + (1 - alpha^2) c^2), the square root shared by the
// functions below (the same quantity as in Lambda, times c)
GFX_HD inline float smith_root(float alpha, float cos_w) {
    const float a2 = alpha * alpha;

    return sqrtf(a2 + (1.0f - a2) * cos_w * cos_w);
}

// Height-correlated Smith G2 for the light direction l and the view
// direction v together: 1 / (1 + Lambda(v) + Lambda(l)). "Correlated"
// means a microfacet that is high up is both less hidden from l and less
// hidden from v, so the two are not independent; this is more accurate
// than the product G1(l) G1(v) of the simpler model. Written without the
// divisions by the cosines (the algebra is
// 2 cl cv / (Sv cl + Sl cv) with S = smith_root), so it stays finite for
// cosines close to 0. At or below the horizon: 0
GFX_HD inline float smith_g2_height_correlated(
                float alpha,
                float n_dot_l,
                float n_dot_v) {

    if (!(n_dot_l > 0.0f) || !(n_dot_v > 0.0f)) {
        return 0.0f;
    }

    const float cl = fminf(n_dot_l, 1.0f);
    const float cv = fminf(n_dot_v, 1.0f);

    return 2.0f * cl * cv /
           (smith_root(alpha, cv) * cl + smith_root(alpha, cl) * cv);
}

// G2 / G1(v), the weight of the visible-normal sampling below, in one
// piece: cl (cv + Sv) / (Sv cl + Sl cv). It is at most 1 (G2 never
// exceeds G1(v)). Needs cv > 0 or cl > 0 to be defined; a direction at
// or below the horizon gives 0
GFX_HD inline float smith_g2_over_g1(
                float alpha,
                float n_dot_l,
                float n_dot_v) {

    if (!(n_dot_l > 0.0f) || !(n_dot_v > 0.0f)) {
        return 0.0f;
    }

    const float cl = fminf(n_dot_l, 1.0f);
    const float cv = fminf(n_dot_v, 1.0f);
    const float sv = smith_root(alpha, cv);
    const float sl = smith_root(alpha, cl);

    return cl * (cv + sv) / (sv * cl + sl * cv);
}

// Schlick's approximation of the Fresnel reflectance, per colour channel:
// f0 at cos_theta = 1 (looking straight on), 1 at cos_theta = 0 (along
// the surface). cos_theta is v.h (the angle at the microfacet that does
// the reflecting) in the specular BRDF, and n.v for the layering below
GFX_HD inline Vec3 fresnel_schlick(Vec3 f0, float cos_theta) {
    const float k = one_minus_pow5(cos_theta);

    return {f0.x + (1.0f - f0.x) * k,
            f0.y + (1.0f - f0.y) * k,
            f0.z + (1.0f - f0.z) * k};
}

GFX_HD inline float fresnel_schlick(float f0, float cos_theta) {
    return f0 + (1.0f - f0) * one_minus_pow5(cos_theta);
}

// The reflectance of the clear dielectric layer on top of the diffuse
// base, one number for all colours. f0 = 0 means "this surface has no such
// layer" (the viewer's M switch turns materials off), and then it
// reflects nothing at any angle: the Schlick curve through f0 = 0 would
// still climb to 1 at glancing angles, which is not what a missing layer
// does
GFX_HD inline float fresnel_dielectric_layer(float f0, float cos_theta) {
    return f0 > 0.0f ? fresnel_schlick(f0, cos_theta) : 0.0f;
}

// ---------------------------------------------------------------------
// The specular BRDF
// ---------------------------------------------------------------------

// Vis = G2 / (4 (n.l) (n.v)) with the height-correlated G2. Using the
// formula of smith_g2_height_correlated this simplifies to
//     0.5 / ((n.l) Sv + (n.v) Sl)
// which has no division by a cosine, so no 0 / 0 for a light that
// touches the horizon. Both cosines must be above 0, otherwise 0
GFX_HD inline float ggx_visibility(
                float alpha,
                float n_dot_l,
                float n_dot_v) {

    if (!(n_dot_l > 0.0f) || !(n_dot_v > 0.0f)) {
        return 0.0f;
    }

    return 0.5f / (n_dot_l * smith_root(alpha, fminf(n_dot_v, 1.0f)) +
                   n_dot_v * smith_root(alpha, fminf(n_dot_l, 1.0f)));
}

// D * Vis, the specular BRDF of a perfectly white mirror-like surface
// (F = 1). The light leaving towards v per unit light arriving from l
GFX_HD inline float ggx_specular_white(
                float alpha,
                float n_dot_l,
                float n_dot_v,
                float n_dot_h) {

    return ggx_d(alpha, n_dot_h) * ggx_visibility(alpha, n_dot_l, n_dot_v);
}

// The full specular BRDF f_s(l, v) = D Vis F per colour channel, for a
// surface whose reflectance at normal incidence is f0
GFX_HD inline Vec3 ggx_specular_brdf(
                float alpha,
                Vec3 f0,
                float n_dot_l,
                float n_dot_v,
                float n_dot_h,
                float v_dot_h) {

    const float dv = ggx_specular_white(alpha, n_dot_l, n_dot_v, n_dot_h);

    return fresnel_schlick(f0, v_dot_h) * dv;
}

// ---------------------------------------------------------------------
// Sampling the specular lobe
//
// To bounce a path off a glossy surface we need a random direction l with
// a known probability density, chosen so that the density cancels as much
// of the BRDF as possible (that is what keeps the noise down).
//
// The obvious choice: pick a microfacet normal h with density D(h) (n.h)
// (the distribution itself) and reflect v in it. It wastes a lot of work:
// at glancing view angles many of the picked microfacets face away from
// the viewer, are hidden behind others, and the reflected l points below
// the surface; those samples are thrown away.
//
// Visible-normal sampling (Heitz 2018) picks h only among the microfacets
// the viewer can see, with density
//     D_v(h) = G1(v) max(0, v.h) D(h) / (n.v)
// (D weighted by how much of each microfacet faces v, and normalised by
// the visible area). Reflecting v in h gives l, and the density of l
// follows from the one of h by the factor 1 / (4 v.h) of a reflection:
//     pdf(l) = D_v(h) / (4 v.h) = G1(v) D(h) / (4 n.v).
// Then, in the Monte Carlo weight f_s (n.l) / pdf with
//     f_s = D G2 F / (4 (n.l) (n.v)),
// D, the 4, the (n.l) and the (n.v) all cancel and what remains is
//     F(v.h) G2(l, v) / G1(v).
// A plain, well-behaved number: at most 1 per colour channel, because
// F <= 1 and G2 <= G1(v). A white mirror therefore never returns more
// light than it receives.
// ---------------------------------------------------------------------

// A random visible microfacet normal h in the frame where the surface
// normal is +Z (Heitz 2018, "Sampling the GGX Distribution of Visible
// Normals", the version that works on the hemisphere). v is the view
// direction in that frame (unit length, v.z > 0), u1 and u2 are uniform
// in [0, 1). The result is a unit vector with h.z > 0.
//
// The idea in one paragraph: squeezing the view direction by alpha in x
// and y (and stretching every normal back at the end) turns the GGX
// microfacets into those of the roughest case, alpha = 1, whose visible
// normals are easy to sample: a uniform point on the disc seen from v,
// lifted onto the hemisphere
GFX_HD inline Vec3 ggx_sample_visible_normal_local(
                float alpha,
                Vec3 v,
                float u1,
                float u2) {

    // The view direction in the stretched space
    const Vec3 vh = normalize(Vec3{alpha * v.x, alpha * v.y, v.z});

    // A basis (t1, t2, vh) of that space
    const float len2 = vh.x * vh.x + vh.y * vh.y;
    const Vec3 t1 = len2 > 0.0f
        ? Vec3{-vh.y, vh.x, 0.0f} * (1.0f / sqrtf(len2))
        : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 t2 = cross(vh, t1);

    // A uniform point on the disc, warped so that the half of the disc
    // facing away from v is squeezed into the visible part (the
    // projected area of the visible normals is the union of a half disc
    // and a half ellipse)
    const float r = sqrtf(u1);
    const float phi = 6.2831853f * u2;
    const float p1 = r * cosf(phi);
    float p2 = r * sinf(phi);
    const float s = 0.5f * (1.0f + vh.z);

    p2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1)) + s * p2;

    // Lift the point onto the hemisphere
    const Vec3 nh = t1 * p1 + t2 * p2 +
                    vh * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1 - p2 * p2));

    // Stretch back. The tiny floor of z keeps h strictly above the
    // horizon of the surface
    return normalize(
        Vec3{alpha * nh.x, alpha * nh.y, fmaxf(nh.z, 1.0e-6f)});
}

// The same in world space: n is the shading normal, v the direction to the
// viewer (both unit length). The cosine n.v is limited from below by
// kMinCosView, so a normal that faces slightly away from the viewer
// behaves like a grazing view. Returns the visible microfacet normal h
// in world space
GFX_HD inline Vec3 ggx_sample_visible_normal(
                Vec3 n,
                Vec3 v,
                float alpha,
                float u1,
                float u2) {

    Vec3 t;
    Vec3 b;
    orthonormal_basis(n, t, b);

    const Vec3 v_local{dot(v, t), dot(v, b),
                       fmaxf(dot(v, n), kMinCosView)};

    const Vec3 h = ggx_sample_visible_normal_local(alpha, v_local, u1, u2);

    return t * h.x + b * h.y + n * h.z;
}

// The mirror image of v in the microfacet h: the direction l a ray that
// arrives along -v leaves in (both unit length, pointing away from the
// surface)
GFX_HD inline Vec3 reflect_about(Vec3 v, Vec3 h) {
    return h * (2.0f * dot(v, h)) - v;
}

// ---------------------------------------------------------------------
// Choosing between the two lobes
//
// A bounce continues in one direction, and the surface has two lobes (the
// diffuse and the specular one). The path tracer picks a lobe at random,
// samples a direction from that lobe alone, and divides the weight by the
// probability of having picked the lobe. That stays unbiased: the
// expected value of "(lobe value / its probability) if picked, else 0" is
// probability * (value / probability) = value for each lobe, and the sum
// over both lobes is the full BRDF. The lobe that was not picked is
// simply never evaluated for the bounce.
//
// The probability should follow how much each lobe matters: if the
// specular lobe reflects 90% of the light, picking it 90% of the time
// spends the samples where the energy is, and the weights stay close to 1
// (less noise). It is limited to 0.1..0.9 so a weak lobe is still sampled
// now and then and the weight of a rare pick stays moderate.
// ---------------------------------------------------------------------

// specular_reflectance: the Fresnel reflectance of the specular lobe at
// the view angle, F(n.v) per colour channel; diffuse_colour: Kd per colour
// channel. Returns the probability of sampling the specular lobe.
// Exactly 0 when the specular lobe reflects nothing, exactly 1 when the
// diffuse lobe does, otherwise inside kSpecularProbabilityMin..Max
GFX_HD inline float specular_lobe_probability(
                Vec3 specular_reflectance,
                Vec3 diffuse_colour) {

    const float w_s = max_component(specular_reflectance);
    const float w_d = max_component(diffuse_colour);

    if (!(w_s > 0.0f)) {
        return 0.0f;
    }

    if (!(w_d > 0.0f)) {
        return 1.0f;
    }

    return fminf(fmaxf(w_s / (w_s + w_d), kSpecularProbabilityMin),
                 kSpecularProbabilityMax);
}

} // namespace gfx
