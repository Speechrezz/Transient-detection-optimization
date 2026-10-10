//
// Created by Mark on 10/1/2026.
//

#include "ExponentialDecay.h"

#include <algorithm>
#include <cmath>

namespace xynth
{

void ExponentialDecay::setSmoothTime(double sampleRate, float timeMs)
{
    constexpr double pi = 3.14159265358979323846;
    float expFactor = static_cast<float>(-2.0 * pi * 1000.0 / sampleRate);

    timeCoefficient = timeMs < 0.001f 
        ? 0.f
        : std::exp(expFactor / timeMs);
}

float ExponentialDecay::processSample(float targetValue) noexcept
{
    const bool isWithinThreshold = std::abs(currentValue - targetValue) < 1e-4f;

    if (isWithinThreshold)
    {
        currentValue = targetValue;
    }
    else
    {
        const float decayValue = targetValue + timeCoefficient * (currentValue - targetValue);
        currentValue = std::max(targetValue, decayValue);
    }

    return currentValue;
}

void ExponentialDecay::processInplace(float* buffer, size_t length) noexcept
{
    for (size_t i = 0; i < length; ++i)
    {
        buffer[i] = processSample(buffer[i]);
    }
}

void ExponentialDecay::processOutplace(const float* in, float* out, size_t length) noexcept
{
    for (size_t i = 0; i < length; ++i)
    {
        out[i] = processSample(in[i]);
    }
}

} // namespace xynth