#pragma once
#include <string>
#include <vector>

using namespace std;

struct Image {
    vector<string> rows;
};

struct Dataset {
    vector<Image> images;
    vector<int> labels;
    int height = 0;
    int width = 0;
    int numClasses = 0;
};
