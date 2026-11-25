#include "Perceptron.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
using namespace std;

Perceptron::Perceptron(int epochs, unsigned seedValue, bool verboseOutput) {
    maxEpochs = epochs;
    seed = seedValue;
    verbose = verboseOutput;
    numClasses = 0;
    numFeatures = 0;
}

// Score of class c for image x:  bias + sum of the weights where the feature is 1
double Perceptron::score(int c, const FeatureVector& x) const {
    double s = weights[c][numFeatures];          // bias is stored in the last slot
    for (int i = 0; i < numFeatures; ++i)
        if (x[i] == 1) s += weights[c][i];
    return s;
}

int Perceptron::predict(const FeatureVector& x) const {
    int bestClass = 0;
    double bestScore = score(0, x);
    for (int c = 1; c < numClasses; ++c) {
        double s = score(c, x);
        if (s > bestScore) {
            bestScore = s;
            bestClass = c;
        }
    }
    return bestClass;
}

void Perceptron::train(const vector<FeatureVector>& X, const vector<int>& y, int nClasses) {
    numClasses = nClasses;
    numFeatures = X[0].size();

    // Start with every weight (and bias) at 0
    weights.assign(numClasses, vector<double>(numFeatures + 1, 0.0));

    // We'll visit the training images in a random order each pass
    mt19937 rng(seed);
    vector<int> order(X.size());
    iota(order.begin(), order.end(), 0);      // order = 0, 1, 2, ..., N-1

    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {
        shuffle(order.begin(), order.end(), rng);
        int mistakes = 0;

        for (int n : order) {
            int guess = predict(X[n]);
            int truth = y[n];

            if (guess != truth) {
                mistakes++;
                for (int i = 0; i < numFeatures; ++i) {
                    if (X[n][i] == 1) {
                        weights[truth][i] += 1.0;   // reward the correct class
                        weights[guess][i] -= 1.0;   // punish the wrong guess
                    }
                }
                weights[truth][numFeatures] += 1.0; // same for the bias
                weights[guess][numFeatures] -= 1.0;
            }
        }

        if (verbose) cout << "  epoch " << epoch << ": " << mistakes << " mistakes\n";
        if (mistakes == 0) break;                   // perfect on training data, stop early
    }
}