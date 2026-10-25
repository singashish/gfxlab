#pragma once

/*
 * The settings of the path tracer that the user can change in the panel
 * (P, "Path tracing" > "Settings"). Header-only and plain C++, so the
 * app, the CUDA renderer and the tests share one definition.
 *
 * The defaults are the constants of core/path_math.h, so a fresh
 * PathTraceSettings reproduces the picture the path tracer had before
 * there were settings. Which changes restart the accumulation (a new
 * picture) and which only change how the existing samples are shown or
 * how long they are collected is decided by the renderer: bounces, sun
 * and sky strength and the firefly limit change the light that is
 * collected, so they restart it; the exposure is applied when the sums
 * are turned into a picture, and the two sample counts only decide when
 * to stop, so they do not.
 */

#include "core/path_math.h"

namespace gfx {

// The ranges the settings are kept in (the sliders use the same ones)
constexpr int kPtMaxSamplesLow = 16;
constexpr int kPtMaxSamplesHigh = 4096;
constexpr int kPtSamplesPerFrameLow = 1;
constexpr int kPtSamplesPerFrameHigh = 16;
constexpr int kPtBouncesLow = 1;
constexpr int kPtBouncesHigh = 12;
constexpr float kPtExposureEvLow = -4.0f;
constexpr float kPtExposureEvHigh = 4.0f;
constexpr float kPtSunStrengthLow = 0.0f;
constexpr float kPtSunStrengthHigh = 20.0f;
constexpr float kPtSkyStrengthLow = 0.0f;
constexpr float kPtSkyStrengthHigh = 4.0f;

struct PathTraceSettings {
    // Stop adding samples at this count per pixel
    int max_samples = static_cast<int>(kPtMaxSamples);

    // New samples per pixel and frame (more = faster convergence, a lower
    // frame rate)
    int samples_per_frame = static_cast<int>(kPtSamplesPerFrame);

    // Surfaces a path may shade; 1 = direct light only
    int max_bounces = kPtMaxBounces;

    // Exposure in stops: the linear light is multiplied by 2^ev (times
    // kPtExposure) before tone mapping
    float exposure_ev = 0.0f;

    // Irradiance of the sun (E_sun); 0 = overcast, only the sky lights
    float sun_strength = kSunIrradiance;

    // Multiplies the sky's light (not the viewport background colour)
    float sky_strength = 1.0f;

    // The limit on rare very bright indirect samples (core/path_math.h,
    // kPtIndirectClamp): on = fewer fireflies, slightly dimmer caustics
    bool firefly_limit = true;
};

namespace detail {

inline int clamp_int_setting(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

// A NaN gives the default; an infinity is clamped to the nearest end of
// the range like any other out-of-range value
inline float clamp_float_setting(float value, float low, float high,
                                 float fallback) {
    if (!(value == value)) {
        return fallback;
    }

    return value < low ? low : (value > high ? high : value);
}

} // namespace detail

// A copy with every field inside its range
inline PathTraceSettings sanitized(const PathTraceSettings& in) {
    const PathTraceSettings defaults;
    PathTraceSettings out = in;

    out.max_samples = detail::clamp_int_setting(
        in.max_samples, kPtMaxSamplesLow, kPtMaxSamplesHigh);
    out.samples_per_frame = detail::clamp_int_setting(
        in.samples_per_frame, kPtSamplesPerFrameLow,
        kPtSamplesPerFrameHigh);
    out.max_bounces = detail::clamp_int_setting(
        in.max_bounces, kPtBouncesLow, kPtBouncesHigh);
    out.exposure_ev = detail::clamp_float_setting(
        in.exposure_ev, kPtExposureEvLow, kPtExposureEvHigh,
        defaults.exposure_ev);
    out.sun_strength = detail::clamp_float_setting(
        in.sun_strength, kPtSunStrengthLow, kPtSunStrengthHigh,
        defaults.sun_strength);
    out.sky_strength = detail::clamp_float_setting(
        in.sky_strength, kPtSkyStrengthLow, kPtSkyStrengthHigh,
        defaults.sky_strength);

    return out;
}

} // namespace gfx
