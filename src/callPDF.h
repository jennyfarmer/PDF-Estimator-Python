/*
 * callPDF.h
 * Standalone (non-MATLAB) entry point for the PDF-Estimator library.
 */

#ifndef CALLPDF_H
#define CALLPDF_H

#include "InputParameters.h"
#include "InputData.h"
#include "Score.h"
#include "ScoreQZ.h"
#include "MinimizeScore.h"
#include "WriteResults.h"
#include "OutputControl.h"
#include "Variable.h"
#include "JointProbability.h"

#include <vector>
#include <string>

using namespace std;

class callPDF {
public:
    callPDF();
    virtual ~callPDF();

    void makeCall(int sampleLength, double* sampleData);
    void makeCallmv(int sampleLength, int variableLength, int pdfSize, double* sampleData);

    void setHigh(double high);
    void setLow(double low);
    void setSURDtarget(int surd);
    void setSURDmin(int surd);
    void setSURDmax(int surd);
    void setLagrangeMin(int lagrange);
    void setLagrangeMax(int lagrange);
    void setDebug(bool debug);

    bool   solutionFailed;
    double confidenceScore;

    vector<double> Vr;
    vector<double> Vcdf;
    vector<double> Vpdf;
    vector<double> Vx;
    vector<int>    VpdfSizes;
    vector<double> Vlagrange;
    vector<double> Vsqr;
    vector<double> Vmarginals;
    vector<double> VxMarginals;
    vector<int> VmarginalSizes;

private:
    InputParameters input;
    OutputControl out;
    bool debug       = false;
};

#endif /* CALLPDF_H */
