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