#pragma once
#include "Classifier.h"

class NaiveBayes : public Classifier {
public:
    NaiveBayes(double smoothing = 1.0);

    void train(const vector<FeatureVector>& X, const vector<int>& y, int numClasses) override;
    int predict(const FeatureVector& x) const override;

private:
    double k;
    int numClasses;
    int numFeatures;
    vector<double> logPrior;
    vector<vector<double>> logOn;
    vector<vector<double>> logOff;
};