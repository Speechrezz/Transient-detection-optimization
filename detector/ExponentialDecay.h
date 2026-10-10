//
// Created by Mark on 10/1/2026.
//

#pragma once

namespace xynth
{

// Based off of juce::dsp::BallisticsFilter.
class ExponentialDecay
{
public:
    void setSmoothTime(double sampleRate, float timeMs);

    float processSample(float sample) noexcept;
    void processInplace(float* buffer, size_t length) noexcept;
    void processOutplace(const float* in, float* out, size_t length) noexcept;

private:
    float timeCoefficient = 0.f;
    float currentValue = 0.f;

};

} // namespace xynth