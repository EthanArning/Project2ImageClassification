#pragma once
#include "Types.h"

// Method 1: one feature per character position (1 = ink, 0 = space)
FeatureVector extractPixel(const Image& img);

// Method 2: one feature per window (1 = window has at least `threshold` ink chars)
FeatureVector extractDensity(const Image& img, int winH, int winW, int threshold);

// Features for ONE image. useDensity=false -> pixel features, true -> density features.
FeatureVector extractOne(const Image& img, bool useDensity);

// Features for EVERY image in a dataset.
vector<FeatureVector> extractAll(const Dataset& data, bool useDensity);

// Picks the best density window size / threshold for "digits" or "faces".
// Call this once, right after loading the data.
void chooseDensitySettings(const string& datasetName);