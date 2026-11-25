#pragma once
#include "Classifier.h"

class Perceptron : public Classifier {
public:
    Perceptron(int epochs = 10, unsigned seed = 1, bool verbose = false);

    void train(const vector<FeatureVector>& X, const vector<int>& y, int numClasses) override;
    int predict(const FeatureVector& x) const override;

private:
    int maxEpochs;
    unsigned seed;
    bool verbose;
    int numClasses;
    int numFeatures;
    vector<vector<double>> weights;   // the LAST slot of each row is the bias

    double score(int c, const FeatureVector& x) const;
};