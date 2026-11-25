#pragma once
#include <string>
#include "Classifier.h"

// Results for one training-set size (e.g. 30% of the training data)
struct PercentResult {
    int percent;
    double meanAcc, stdAcc;          // accuracy over the trials (0 to 1)
    double meanTimeMs, stdTimeMs;    // training time in milliseconds
};

// Fraction of items in X that the model labels correctly.
double accuracy(const Classifier& model, const vector<FeatureVector>& X, const vector<int>& y);

// For 10%, 20%, ..., 100%: repeat `trials` times { random subset -> train (timed) -> test }.
vector<PercentResult> runExperiment(Classifier& model,
                                    const vector<FeatureVector>& Xtrain, const vector<int>& ytrain,
                                    const vector<FeatureVector>& Xtest,  const vector<int>& ytest,
                                    int numClasses, int trials);

void printResults(const string& title, const vector<PercentResult>& results);
void writeCsv(const string& path, const vector<PercentResult>& results);