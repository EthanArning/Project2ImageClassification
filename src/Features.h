#pragma once
#include "Types.h"

// Method 1: one feature per character position (1 = ink, 0 = space)
FeatureVector extractPixel(const Image& img);

// Method 2: one feature per window (1 = window has at least `threshold` ink chars)
FeatureVector extractDensity(const Image& img, int winH, int winW, int threshold);