//
// Created by Mark on 10/8/2026.
//

#pragma once

namespace xynth
{

// Useful for smoothing parameter values. Based off of juce::dsp::BallisticsFilter.
class ExponentialSmoother
{
public:
    void setSmoothTime(double sampleRate, float attackTimeMs, float decayTimeMs);

    float processSample(float sample) noexcept;
    void processInplace(float* buffer, size_t length) noexcept;
    void processOutplace(const float* in, float* out, size_t length) noexcept;

private:
    float attackCoefficient = 0.f, decayCoefficient = 0.f;;
    float currentValue = 0.f;

};

} // namespace xynth