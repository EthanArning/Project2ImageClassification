#include "Features.h"
#include <algorithm>
using namespace std;

// ---------- Method 1: raw pixel features ----------
FeatureVector extractPixel(const Image& img) {
    FeatureVector features;
    for (const string& row : img.rows) {
        for (char c : row) {
            if (c != ' ') features.push_back(1);   // '+' or '#'  -> ink
            else          features.push_back(0);   // ' '         -> background
        }
    }
    return features;
}

// ---------- Method 2: binary local density features ----------
FeatureVector extractDensity(const Image& img, int winH, int winW, int threshold) {
    FeatureVector features;
    int H = img.rows.size();
    int W = img.rows[0].size();

    for (int top = 0; top < H; top += winH) {
        for (int left = 0; left < W; left += winW) {
            int inkCount = 0;
            int bottom = min(top + winH, H);
            int right  = min(left + winW, W);
            for (int r = top; r < bottom; ++r)
                for (int c = left; c < right; ++c)
                    if (img.rows[r][c] != ' ') inkCount++;

            if (inkCount >= threshold) features.push_back(1);
            else                       features.push_back(0);
        }
    }
    return features;
}

// ---------- Helpers: pick a method for one image / a whole dataset ----------
// The density settings live HERE so there is exactly one place to change them.
FeatureVector extractOne(const Image& img, bool useDensity) {
    if (useDensity) return extractDensity(img, 4, 4, 3);
    return extractPixel(img);
}

vector<FeatureVector> extractAll(const Dataset& data, bool useDensity) {
    vector<FeatureVector> all;
    for (const Image& img : data.images)
        all.push_back(extractOne(img, useDensity));
    return all;
}

// The density settings live HERE so there is exactly one place to change them.
// They were chosen by trying many values and comparing VALIDATION accuracy.
static int densityWinH = 4, densityWinW = 4, densityThreshold = 3;

void chooseDensitySettings(const string& datasetName) {
    if (datasetName == "digits") {
        densityWinH = 2; densityWinW = 2; densityThreshold = 1;
    } else if (datasetName == "faces") {
        densityWinH = 4; densityWinW = 4; densityThreshold = 2;
    }
}

FeatureVector extractOne(const Image& img, bool useDensity) {
    if (useDensity) return extractDensity(img, densityWinH, densityWinW, densityThreshold);
    return extractPixel(img);
}