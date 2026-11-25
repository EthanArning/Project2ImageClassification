#include "NaiveBayes.h"
#include <cmath>
using namespace std;

NaiveBayes::NaiveBayes(double smoothing) {
    k = smoothing;
    numClasses = 0;
    numFeatures = 0;
}

void NaiveBayes::train(const vector<FeatureVector>& X, const vector<int>& y, int nClasses) {
    numClasses = nClasses;
    numFeatures = X[0].size();

    // ---- Step A: count ----
    vector<int> classCount(numClasses, 0);
    vector<vector<int>> onCount(numClasses, vector<int>(numFeatures, 0));

    for (size_t n = 0; n < X.size(); ++n) {
        int c = y[n];
        classCount[c]++;
        for (int i = 0; i < numFeatures; ++i)
            onCount[c][i] += X[n][i];
    }

    // ---- Step B: turn counts into log probabilities ----
    logPrior.assign(numClasses, 0.0);
    logOn.assign(numClasses, vector<double>(numFeatures, 0.0));
    logOff.assign(numClasses, vector<double>(numFeatures, 0.0));

    for (int c = 0; c < numClasses; ++c) {
        logPrior[c] = log((classCount[c] + k) / (X.size() + k * numClasses));
        for (int i = 0; i < numFeatures; ++i) {
            double pOn = (onCount[c][i] + k) / (classCount[c] + 2.0 * k);
            logOn[c][i]  = log(pOn);
            logOff[c][i] = log(1.0 - pOn);
        }
    }
}

int NaiveBayes::predict(const FeatureVector& x) const {
    int bestClass = 0;
    double bestScore = -1e300;

    for (int c = 0; c < numClasses; ++c) {
        double score = logPrior[c];
        for (int i = 0; i < numFeatures; ++i) {
            if (x[i] == 1) score += logOn[c][i];
            else           score += logOff[c][i];
        }
        if (score > bestScore) {
            bestScore = score;
            bestClass = c;
        }
    }
    return bestClass;
}