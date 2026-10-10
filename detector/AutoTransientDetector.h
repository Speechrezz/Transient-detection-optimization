//
// Created by Mark on 6/11/2026.
//

#pragma once

#include "ExponentialDecay.h"
#include "ExponentialSmoother.h"

namespace xynth
{

struct TransientParams
{
    double sampleRate;
    float fastDecay;
    float slowDecay;
    float slowAttack;
    float triggerThresh;
    float rearmRatio;
    float noiseFloor = 0.0005; // -66 dB
};

class AutoTransientDetector
{
public:
    AutoTransientDetector() = default;

    void prepare(TransientParams) noexcept;
    void process(const float* input, float* output, size_t length) noexcept;

protected:
    TransientParams params;

    ExponentialDecay envelopeFast;
    ExponentialSmoother envelopeSlow;

    bool isArmed = true;

};

}