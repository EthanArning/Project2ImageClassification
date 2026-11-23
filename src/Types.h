#pragma once
#include <string>
#include <vector>
using namespace std;

// One ASCII image: a list of text rows, e.g. 28 strings of length 28.
struct Image {
    vector<string> rows;
};

// A whole file (train, validation or test): many images + their labels.
struct Dataset {
    vector<Image> images;
    vector<int> labels;
    int height = 0;
    int width = 0;
    int numClasses = 0;   // 10 for digits
};

// The list of 0/1 numbers describing ONE image.
using FeatureVector = vector<int>;