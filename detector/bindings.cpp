#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "AutoTransientDetector.h"

namespace py = pybind11;

class DetectorWrapper
{
public:
    DetectorWrapper() = default;

    void prepare(xynth::TransientParams params)
    {
        detector.prepare(params);
    }
    py::array_t<float> process(py::array_t<float, py::array::c_style> audio)
    {
        py::array_t<float> output(audio.size());
        detector.process(audio.data(), output.mutable_data(), audio.size());

        return output;
    }

protected:
    xynth::AutoTransientDetector detector;

};

PYBIND11_MODULE(detector_cpp, m) {
    m.doc() = "Transient detector";

    py::class_<xynth::TransientParams>(m, "TransientParamsCpp")
        .def(py::init<>()) // Constructor
        .def(py::init<double, float, float, float, float, float>(),
            py::arg("sample_rate"),
            py::arg("fast_decay"),
            py::arg("slow_decay"),
            py::arg("slow_attack"),
            py::arg("trigger_thresh"),
            py::arg("rearm_ratio")
        )
        .def_readwrite("sample_rate",    &xynth::TransientParams::sampleRate)
        .def_readwrite("fast_decay",     &xynth::TransientParams::fastDecay)
        .def_readwrite("slow_decay",     &xynth::TransientParams::slowDecay)
        .def_readwrite("slow_attack",    &xynth::TransientParams::slowAttack)
        .def_readwrite("trigger_thresh", &xynth::TransientParams::triggerThresh)
        .def_readwrite("rearm_ratio",    &xynth::TransientParams::rearmRatio)
        .def_readwrite("noise_floor",    &xynth::TransientParams::noiseFloor);

    py::class_<DetectorWrapper>(m, "TransientDetectorCpp")
        .def(py::init<>()) // Constructor
        .def("prepare", &DetectorWrapper::prepare)
        .def("process", &DetectorWrapper::process);
}