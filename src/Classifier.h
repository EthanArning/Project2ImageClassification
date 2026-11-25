#pragma once
#include "Types.h"

// A "contract": anything that is a Classifier must be able to train and predict.
class Classifier {
public:
    virtual ~Classifier() {}
    virtual void train(const vector<FeatureVector>& X, const vector<int>& y, int numClasses) = 0;
    virtual int predict(const FeatureVector& x) const = 0;
};