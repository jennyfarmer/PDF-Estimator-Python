/*
 * bindings.cpp
 * pybind11 bindings for the PDF-Estimator C++ library.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "callPDF.h"
#include <iostream>

namespace py = pybind11;

py::dict estimate_pdf(
    py::array_t<double, py::array::c_style | py::array::forcecast> data,
    bool   debug = false
) {
    callPDF pd;
    pd.setDebug(debug);

    auto buf = data.request();
    double* ptr = static_cast<double*>(buf.ptr);

    int variableLength = buf.shape[0];
    int sampleLength = buf.shape[1];

    cout << "variableLength = " << variableLength << "\n";
    cout << "sampleLength = " << sampleLength << "\n";

    if (variableLength > 1) {
        pd.makeCallmv(sampleLength, variableLength, ptr);
    }
    else {
        pd.makeCall(sampleLength, ptr);
    }

    py::dict result;
    result["y"]          = py::array_t<double>(pd.Vx.size(),       pd.Vx.data());
    result["pdf"]        = py::array_t<double>(pd.Vpdf.size(),     pd.Vpdf.data());
    result["pdfSizes"]   = py::array_t<int>(pd.VpdfSizes.size(),   pd.VpdfSizes.data());
    result["cdf"]        = py::array_t<double>(pd.Vcdf.size(),     pd.Vcdf.data());
    result["sqr"]        = py::array_t<double>(pd.Vsqr.size(),     pd.Vsqr.data());
    result["lagrange"]   = py::array_t<double>(pd.Vlagrange.size(),pd.Vlagrange.data());
    result["confidence"] = pd.confidenceScore;
    result["SURD"]       = py::array_t<double>(pd.Vr.size(),         pd.Vr.data());
    result["marginals"]  = py::array_t<double>(pd.Vmarginals.size(), pd.Vmarginals.data());
    result["marginalsY"] = py::array_t<double>(pd.VxMarginals.size(), pd.VxMarginals.data());
    result["marginalSizes"] = py::array_t<int>(pd.VmarginalSizes.size(), pd.VmarginalSizes.data());
    result["failed"] = (int)pd.solutionFailed;
    return result;
}

PYBIND11_MODULE(_pdf_estimator, m) {
    m.doc() = "Python bindings for the PDF-Estimator C++ library (Farmer & Jacobs 2019).";
    m.def("estimate_pdf", &estimate_pdf,
        py::arg("data"),
        py::arg("debug") = false
    );
}
