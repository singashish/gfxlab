#pragma once

/*
 * The maths of the CUDA path tracer ("Rendered" mode), kept in a portable
 * header so the CPU tests run the very same code as the GPU. Header-only
 * and GFX_HD like raster_math.h and ray.h; nothing here calls a host-only
 * std:: function (only sqrtf, powf, cosf, ...), so nvcc accepts it in
 * device code.
 *
 * The ideas, in the order the renderer uses them:
 *
 * Linear light. Light adds up and scales in a physical way only when the
 * numbers are proportional to energy ("linear"). Colours stored in files
 * and shown on a monitor are sRGB: a curve that spends more numbers on
 * dark tones. So the path tracer decodes every stored colour to linear
 * (decode_srgb), does all of its arithmetic there, and only at the very
 * end turns the result into a picture (tonemap_aces, then encode_srgb).
 * Mixing the two (adding sRGB numbers) would make everything too dark or
 * too bright and break the energy balance that makes lighting look right.
 *
 * Monte Carlo. The light arriving at a point is an integral over all
 * directions that cannot be solved in closed form for a real scene. We
 * estimate it: pick a random direction, measure what comes from there,
 * divide by the probability of having picked it. The average of many such
 * single-sample estimates converges to the true value (the estimator is
 * "unbiased"), with noise that falls like 1 / sqrt(number of samples).
 *
 * Cosine sampling. A diffuse (Lambert) surface with albedo a reflects
 * a/pi * cos(theta) of the light from a direction at angle theta from the
 * normal. If directions are chosen with probability density cos(theta)/pi
 * (more often near the normal), that density cancels the whole factor and
 * one bounce simply multiplies the path's "throughput" by a: no cosines,
 * no pi, no division left in the loop (see cosine_hemisphere).
 *
 * Glossy surfaces. A surface is a mix of this diffuse layer and a glossy
 * (GGX) specular lobe: the next direction is sampled from one of the two,
 * and the sun is evaluated for both. The maths of the glossy lobe has its
 * own portable header, brdf.h.
 *
 * Next-event estimation. The sun covers only a tiny part of the sky (a
 * cone of 1 degree half angle). A random bounce direction would hit it
 * about once in 10000 tries, so the picture would be almost all noise. So
 * at every surface hit we aim one extra "shadow ray" straight at a random
 * point of the sun's disc and add its light explicitly. To avoid counting
 * the sun twice, a bounce ray that happens to hit the sun by chance adds
 * nothing: the sun is always handled by the explicit ray, never by chance.
 * The sun and the sky are only lights: no ray that leaves the camera and
 * misses everything shows them, the viewport background is drawn there
 * instead (see background_radiance), so only their effect is visible.
 *
 * Russian roulette. Paths cannot be followed forever, but cutting them off
 * at a fixed length loses energy. Instead, from some bounce on, a path
 * survives with probability q and its throughput is divided by q. The
 * expected value is q * (x / q) + (1 - q) * 0 = x: the average stays the
 * same (still unbiased), only fewer rays are traced. Dim paths (small
 * throughput, little to contribute) get small q.
 */

#include "core/common.h"
#include "core/math_util.h"
#include "core/vec3.h"

#include <cmath>

namespace gfx {

// ---------------------------------------------------------------------
// Tunable constants
// ---------------------------------------------------------------------

// A path hits and shades at most this many surfaces (1 = direct light
// only; the light a surface passes on to the next one is what the extra
// bounces add: colour bleeding, brighter shadows)
constexpr int kPtMaxBounces = 4;

// New samples per pixel and frame. Every frame is one more independent
// sample, averaged into the accumulation buffer
constexpr unsigned int kPtSamplesPerFrame = 1;

// After this many samples per pixel the picture counts as converged and
// the path tracing kernel stops until something changes
constexpr unsigned int kPtMaxSamples = 1024;

// Linear light is multiplied by this before tone mapping
constexpr float kPtExposure = 1.0f;

// Russian roulette starts at this bounce (counted from 0 = camera hit)
constexpr int kPtRouletteStart = 3;

// The sun is a disc of light seen from the surfaces under this half angle
// (the real sun is about 0.27 degrees; a larger disc gives softer
// shadows). The disc is never drawn in the picture, only its light is
constexpr float kSunHalfAngle = 1.0f * (kPi / 180.0f);

// Irradiance the sun delivers to a surface facing it, in the renderer's
// units (a white surface facing the sun reflects E / pi = 1.6 before
// tone mapping)
constexpr float kSunIrradiance = 5.0f;

// Overall brightness of the sky colours below
constexpr float kSkyScale = 0.9f;

// Directions this far below the horizon (measured as dir.z) are fully
// the ground colour; the colour fades in over this band, so there is no
// hard line at the horizon
constexpr float kHorizonBand = 0.02f;

// Survival probability limits of Russian roulette (see below)
constexpr float kRouletteMinSurvival = 0.05f;
constexpr float kRouletteMaxSurvival = 0.95f;

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

GFX_HD inline float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

// True for ordinary numbers, false for NaN and +/-infinity. (NaN fails
// every comparison, so it falls out of the <=.)
GFX_HD inline bool finite_float(float v) {
    return fabsf(v) <= 3.402823466e38f;
}

GFX_HD inline bool finite_vec3(Vec3 v) {
    return finite_float(v.x) && finite_float(v.y) && finite_float(v.z);
}

// Component-wise product (colour times colour)
GFX_HD inline Vec3 modulate(Vec3 a, Vec3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

// ---------------------------------------------------------------------
// sRGB <-> linear (the exact piecewise standard curve)
//
// Colours of materials and textures are stored as display (sRGB)
// values; the path tracer works in linear light. One rule for both:
// stored colour -> decode_srgb -> linear albedo. Inputs outside 0..1
// are clamped, so an albedo can never exceed 1 (a surface cannot
// reflect more light than it receives)
// ---------------------------------------------------------------------

GFX_HD inline float decode_srgb(float s) {
    s = clamp01(s);

    return s <= 0.04045f
        ? s / 12.92f
        : powf((s + 0.055f) / 1.055f, 2.4f);
}

GFX_HD inline float encode_srgb(float l) {
    l = clamp01(l);

    return l <= 0.0031308f
        ? l * 12.92f
        : 1.055f * powf(l, 1.0f / 2.4f) - 0.055f;
}

GFX_HD inline Vec3 decode_srgb(Vec3 s) {
    return {decode_srgb(s.x), decode_srgb(s.y), decode_srgb(s.z)};
}

GFX_HD inline Vec3 encode_srgb(Vec3 l) {
    return {encode_srgb(l.x), encode_srgb(l.y), encode_srgb(l.z)};
}

// ---------------------------------------------------------------------
// Tone mapping
//
// Linear light has no upper limit (the sun is far brighter than a lit
// wall), but a monitor shows 0..1. Cutting off at 1
// would turn every bright area into a flat white patch. A filmic curve
// compresses instead: nearly linear for dark tones, a soft shoulder for
// bright ones, so highlights roll off smoothly. This is the ACES fit by
// Krzysztof Narkowicz, applied per channel
// ---------------------------------------------------------------------

GFX_HD inline float tonemap_aces(float x) {
    // The curve is above 1 for any x > about 8, and squaring huge values
    // would overflow to infinity (infinity / infinity is NaN), so
    // limit the input first
    x = fminf(fmaxf(x, 0.0f), 1.0e4f);

    const float y = (x * (2.51f * x + 0.03f)) /
                    (x * (2.43f * x + 0.59f) + 0.14f);

    return clamp01(y);
}

GFX_HD inline Vec3 tonemap_aces(Vec3 c) {
    return {tonemap_aces(c.x), tonemap_aces(c.y), tonemap_aces(c.z)};
}

// Accumulated mean radiance (linear) to display colour in 0..1: exposure,
// tone mapping, then the sRGB curve. A NaN or infinite input gives black,
// so one bad value can never show up as a bright or flickering pixel
GFX_HD inline Vec3 linear_to_display(Vec3 radiance, float exposure) {
    if (!finite_vec3(radiance)) {
        return {0.0f, 0.0f, 0.0f};
    }

    return encode_srgb(tonemap_aces(radiance * exposure));
}

// The inverse of tonemap_aces: the linear value x that the curve maps to
// y. Rearranging y = x (2.51 x + 0.03) / (x (2.43 x + 0.59) + 0.14) gives
// the quadratic a x^2 + b x + c = 0 with a = 2.43 y - 2.51,
// b = 0.59 y - 0.03 and c = 0.14 y. Its one non-negative root is used,
// written in the form that never subtracts two nearly equal numbers
// (for b < 0 the other form of the same root). The curve only reaches 1
// at infinity, so y is limited to 0..0.999 (a finite answer)
GFX_HD inline float tonemap_aces_inverse(float y) {
    y = fminf(fmaxf(y, 0.0f), 0.999f);

    const float a = 2.43f * y - 2.51f;  // negative for every y < 1
    const float b = 0.59f * y - 0.03f;
    const float c = 0.14f * y;
    const float root = sqrtf(b * b - 4.0f * a * c);

    return b >= 0.0f ? (b + root) / (-2.0f * a) : 2.0f * c / (root - b);
}

GFX_HD inline Vec3 tonemap_aces_inverse(Vec3 y) {
    return {tonemap_aces_inverse(y.x), tonemap_aces_inverse(y.y),
            tonemap_aces_inverse(y.z)};
}

// The linear radiance that linear_to_display turns back into the given
// display colour (0..1 per channel): the path tracer adds it for camera
// rays that miss everything, so the viewport background stays exactly
// the background colour, and a pixel half covered by an object blends
// the object with the background like any other radiance. The three
// steps are the display pipeline run backwards: sRGB decode, inverse
// tone map, divide by the exposure. Display values above about 0.999
// (after decoding) cannot be reached by the tone curve and come back as
// the nearest value it can reach
GFX_HD inline Vec3 background_radiance(Vec3 display, float exposure) {
    const Vec3 radiance = tonemap_aces_inverse(decode_srgb(display));

    return radiance * (1.0f / exposure);
}

// ---------------------------------------------------------------------
// The sky: a light, never shown. Rays that bounce off a surface and miss
// everything collect it, so it lights the scene from all sides. Camera
// rays that miss get the viewport background instead
//
// Nothing here is a texture or a table; the sky is a formula of the
// direction. Z is up. With up = clamp(dir.z, 0, 1), the weight of the
// horizon colour is h = (1 - up)^3 and the colour is
//     horizon * h + zenith * (1 - h)
// (a convex combination, so up = 1 gives exactly the zenith colour and
// up = 0 exactly the horizon colour). The cube makes the bright
// horizon band narrow and the blue deep: h falls from 1 to 0.125 within
// the first 30 degrees of elevation. It is smooth and strictly
// monotone in elevation.
// Below the horizon there is no geometry (the grid is only an overlay),
// so the sky continues as a flat ground colour, faded in over a band of
// kHorizonBand so no hard line appears
// ---------------------------------------------------------------------

GFX_HD inline Vec3 sky_zenith_radiance() {
    return {0.30f * kSkyScale, 0.50f * kSkyScale, 0.95f * kSkyScale};
}

GFX_HD inline Vec3 sky_horizon_radiance() {
    return {0.85f * kSkyScale, 0.88f * kSkyScale, 0.95f * kSkyScale};
}

GFX_HD inline Vec3 sky_ground_radiance() {
    return {0.12f, 0.12f, 0.13f};
}

// dir is a unit vector
GFX_HD inline Vec3 sky_radiance(Vec3 dir) {
    const float up = clamp01(dir.z);
    const float f = 1.0f - up;
    const float h = f * f * f;

    const Vec3 zenith = sky_zenith_radiance();
    const Vec3 horizon = sky_horizon_radiance();
    const Vec3 sky = horizon * h + zenith * (1.0f - h);

    // 0 at and above the horizon, 1 once the band is crossed, smooth
    // (smoothstep) in between
    const float depth = clamp01(-dir.z / kHorizonBand);
    const float g = depth * depth * (3.0f - 2.0f * depth);

    return sky * (1.0f - g) + sky_ground_radiance() * g;
}

// ---------------------------------------------------------------------
// Bases and direction sampling
// ---------------------------------------------------------------------

// Two unit tangents t and b so that (t, b, n) is an orthonormal basis,
// for a unit vector n (Duff et al. 2017, branchless: no special case
// for n close to the z axis)
GFX_HD inline void orthonormal_basis(Vec3 n, Vec3& t, Vec3& b) {
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;

    t = Vec3{1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x};
    b = Vec3{c, sign + n.y * n.y * a, -n.y};
}

// A random direction above the surface with normal n (unit length),
// cosine-weighted: directions near the normal are picked more often,
// matching how much each direction contributes to ambient light.
// u1 and u2 are uniform random numbers in [0, 1). The density is
// cos(theta) / pi, which is exactly what cancels in a Lambert bounce
GFX_HD inline Vec3 cosine_hemisphere(Vec3 n, float u1, float u2) {
    Vec3 t;
    Vec3 bt;
    orthonormal_basis(n, t, bt);

    // A uniform point on a disc, lifted onto the hemisphere
    const float r = sqrtf(u1);
    const float phi = 6.2831853f * u2;

    return t * (r * cosf(phi)) +
           bt * (r * sinf(phi)) +
           n * sqrtf(fmaxf(0.0f, 1.0f - u1));
}

// ---------------------------------------------------------------------
// The sun
// ---------------------------------------------------------------------

// cos of the sun's half angle, the form sample_sun_cone takes
GFX_HD inline float sun_cone_cos(float half_angle) {
    return cosf(half_angle);
}

// A direction uniformly distributed over the sun's cone (uniform in
// solid angle: every part of the disc is equally likely). Along the
// cone's axis the area element is flat in cos(theta), so cos(theta) is
// uniform between cos_half and 1; phi is uniform around the axis.
// sun_dir is a unit vector; u1 and u2 are uniform in [0, 1)
GFX_HD inline Vec3 sample_sun_cone(
                Vec3 sun_dir,
                float cos_half,
                float u1,
                float u2) {

    // Computed as 1 - cos(theta) so the small angles of a narrow cone
    // keep their precision: 1 - cos^2 = (1 - cos) (1 + cos)
    const float one_minus_cos = u1 * (1.0f - cos_half);
    const float cos_theta = 1.0f - one_minus_cos;
    const float sin_theta =
        sqrtf(fmaxf(0.0f, one_minus_cos * (2.0f - one_minus_cos)));
    const float phi = 6.2831853f * u2;

    Vec3 t;
    Vec3 b;
    orthonormal_basis(sun_dir, t, b);

    return t * (sin_theta * cosf(phi)) +
           b * (sin_theta * sinf(phi)) +
           sun_dir * cos_theta;
}

// ---------------------------------------------------------------------
// Russian roulette
// ---------------------------------------------------------------------

// Probability that a path with this throughput continues. Paths that
// still carry a lot of light (the largest colour channel near 1)
// nearly always survive; dim ones are cut off often, but never with
// certainty (the minimum) and never spared with certainty (the
// maximum), so the division by q afterwards stays bounded
GFX_HD inline float russian_roulette_survival(Vec3 throughput) {
    const float largest =
        fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

    return fminf(fmaxf(largest, kRouletteMinSurvival),
                 kRouletteMaxSurvival);
}

} // namespace gfx
