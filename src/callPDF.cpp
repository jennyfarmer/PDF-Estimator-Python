/*
 * callPDF.cpp
 * Standalone entry point – no MATLAB engine dependency.
 */

#include "callPDF.h"

callPDF::callPDF() {
    InputParameters input;
    out.debug = debug;
}

callPDF::~callPDF() {
//    delete input;
}

void callPDF::setLow(double low) {
    input.lowerBoundSpecified = true;
    input.lowerBound = (float)low;
}

void callPDF::setHigh(double high) {
    input.upperBoundSpecified = true;
    input.upperBound = (float)high;
}

void callPDF::setSURDtarget(int surd) { input.SURDTarget  = surd; }
void callPDF::setSURDmin(int surd)    { input.SURDMinimum = surd; }
void callPDF::setSURDmax(int surd)    { input.SURDMaximum = surd; }
void callPDF::setDebug(bool debug)    { this->debug = debug; out.debug = debug; }
void callPDF::setLagrangeMin(int l)   { input.minLagrange = l; }
void callPDF::setLagrangeMax(int l)   { input.maxLagrange = l; }

void callPDF::makeCall(int sampleLength, double* sampleData) {
    solutionFailed = false;

    MinimizeScore minimumPDF = MinimizeScore();
    InputData     data = InputData(input);

    data.out.debug = debug;
    minimumPDF.out.debug = debug;
    input.out.debug = debug;

    // Load data
    vector<double> inputData(sampleData, sampleData + sampleLength);
    input.writeHeader = false;
    input.writeFile = false;
    if (input.initPartitionSize == 0) {
        input.initPartitionSize = sampleLength;
    }
    data.setData(inputData);

    if (data.processData()) {
        solutionFailed = minimumPDF.minimize(input, data);
        if (!solutionFailed) {
            WriteResults write;
            write.out.debug = debug;
            write.createSolution(input, data, minimumPDF);
            write.createQQ(minimumPDF.bestRandom, minimumPDF.N);

            confidenceScore = minimumPDF.bestThreshold;
            Vr = write.R;
            Vlagrange = write.L;
            Vsqr = write.SQR;
            Vcdf = write.CDF;
            Vpdf = write.PDF;
            Vx = write.x;
        }
    }
    else {
        solutionFailed = true;
    }
}

void callPDF::makeCallmv(int sampleLength, int variableLength, double* sampleData) {

    cout << sampleData[0] << "\n";
    cout << sampleData[1000] << "\n";

    int matrixSize = sampleLength * variableLength;
    int pdfSize;
//    pdfSize = ceil(pow(sampleLength / 100.0, 1.0 / (variableLength - 1)));
    pdfSize = pow(sampleLength, 1.0 / variableLength);

    cout << "pdf " << pdfSize << "\n";
    // soften growth (prevents explosion)
    pdfSize *= 0.75;
    cout << "pdf " << pdfSize << "\n";

    if (pdfSize > 100) {
        pdfSize = 100;
    }
    if (pdfSize < 32) {
        pdfSize = 32;
//        if (pdfSize < 2) {
//            pdfSize = 2;
    }

    cout << "pdf " << pdfSize << "\n";
    vector <Variable> variables;
    variables.reserve(variableLength);
    InputParameters input;
    input.out.debug = debug;

    int iVariable = 0;
    vector <double> samples;
    samples.reserve(sampleLength);
    int vSample = 0;
    for (int i = 0; i < matrixSize; i++) {
        samples.push_back(sampleData[i]);
        if (++vSample == sampleLength) {
            ostringstream vString;
            vString << iVariable++;
            Variable variable = Variable(input, samples, vString.str(), false);
            variable.out.debug = debug;
            variables.push_back(variable);
            samples.clear();
            vSample = 0;
        }
    }

    JointProbability jp = JointProbability(variables, sampleLength, pdfSize);
    jp.out.debug = debug;
    jp.calculate();

    Vpdf = jp.getJP();
    Vx = jp.getRange();
    VpdfSizes = jp.getPDFSizes();
    Vmarginals = jp.getMarginals();
    VxMarginals = jp.getMarginalRanges();
    VmarginalSizes = jp.getMarginalSizes();
//    cout << "marginal " << Vmarginals[0] << "\n";
    solutionFailed = false;

}

