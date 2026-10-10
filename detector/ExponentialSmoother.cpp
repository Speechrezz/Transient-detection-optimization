//
// Created by Mark on 10/8/2026.
//

#include "ExponentialSmoother.h"

#include <algorithm>
#include <cmath>

namespace xynth
{

void ExponentialSmoother::setSmoothTime(double sampleRate, float attackTimeMs, float decayTimeMs)
{
    constexpr double pi = 3.14159265358979323846;
    float expFactor = static_cast<float>(-2.0 * pi * 1000.0 / sampleRate);

    attackCoefficient = attackTimeMs < 0.01f
        ? 0.f
        : std::exp(expFactor / attackTimeMs);

    decayCoefficient = decayTimeMs < 0.01f
        ? 0.f
        : std::exp(expFactor / decayTimeMs);
}

float ExponentialSmoother::processSample(float targetValue) noexcept
{
    const bool isWithinThreshold = std::abs(currentValue - targetValue) < 1e-4f;

    if (isWithinThreshold)
    {
        currentValue = targetValue;
    }
    else
    {
        const float timeCoef = targetValue > currentValue
            ? attackCoefficient
            : decayCoefficient;

        currentValue = targetValue + timeCoef * (currentValue - targetValue);
    }

    return currentValue;
}

void ExponentialSmoother::processInplace(float* buffer, size_t length) noexcept
{
    for (size_t i = 0; i < length; ++i)
    {
        buffer[i] = processSample(buffer[i]);
    }
}

void ExponentialSmoother::processOutplace(const float* in, float* out, size_t length) noexcept
{
    for (size_t i = 0; i < length; ++i)
    {
        out[i] = processSample(in[i]);
    }
}

} // namespace xynth