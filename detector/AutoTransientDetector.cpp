//
// Created by Mark on 6/11/2026.
//

#include "AutoTransientDetector.h"

#include <cmath>
#include <algorithm>

namespace xynth
{

void AutoTransientDetector::prepare(TransientParams newParams) noexcept
{
    params = newParams;

    envelopeFast.setSmoothTime(params.sampleRate, params.fastDecay);
    envelopeSlow.setSmoothTime(params.sampleRate, params.slowAttack, params.slowDecay);

    isArmed = true;
}

void AutoTransientDetector::process(const float* input, float* output, size_t length) noexcept
{
    const float highThresh = params.triggerThresh;
    const float lowThresh  = params.triggerThresh * params.rearmRatio;

    for (size_t i = 0; i < length; ++i)
    {
        const float absSample = std::abs(input[i]);

        const float sampleFast = envelopeFast.processSample(absSample);
        const float sampleSlow = envelopeSlow.processSample(absSample);

        const float strength = std::max(0.0f, (sampleFast - sampleSlow) / (sampleSlow + params.noiseFloor));

        float isTransient = 0.f;

        if (isArmed && strength > highThresh)
        {
            isArmed = false;
            isTransient = 1.f;
        }
        else if (!isArmed && strength < lowThresh)
        {
            isArmed = true;
        }

        output[i] = isTransient;
    }
}

}
