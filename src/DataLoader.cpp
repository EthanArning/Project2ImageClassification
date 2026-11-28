#include "DataLoader.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>
using namespace std;

Dataset loadDataset(const string& imagePath, const string& labelPath) {
    Dataset data;

    // Step 1: read labels (one integer per image)
    ifstream labelFile(labelPath);
    if (!labelFile) throw runtime_error("Cannot open " + labelPath);
    int label;
    while (labelFile >> label) data.labels.push_back(label);

    // Step 2: read every line of the image file
    ifstream imageFile(imagePath);
    if (!imageFile) throw runtime_error("Cannot open " + imagePath);
    vector<string> lines;
    string line;
    while (getline(imageFile, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }

    // Step 3: work out the image dimensions
    size_t n = data.labels.size();
    data.height = lines.size() / n;
    for (const auto& l : lines)
        data.width = max(data.width, static_cast<int>(l.size()));

    // Step 4: cut the lines into images
    for (size_t i = 0; i < n; ++i) {
        Image img;
        for (int r = 0; r < data.height; ++r) {
            string row = lines[i * data.height + r];
            row.resize(data.width, ' ');
            img.rows.push_back(row);
        }
        data.images.push_back(img);
    }

    data.numClasses = *max_element(data.labels.begin(), data.labels.end()) + 1;
    return data;
}

// ---------- Where each dataset's files live ----------
bool loadDatasets(const string& name, Dataset& train, Dataset& val, Dataset& test) {
    if (name == "digits") {
        train = loadDataset("data/digitdata/trainingimages",   "data/digitdata/traininglabels");
        val   = loadDataset("data/digitdata/validationimages", "data/digitdata/validationlabels");
        test  = loadDataset("data/digitdata/testimages",       "data/digitdata/testlabels");
        return true;
    }
    if (name == "faces") {
        train = loadDataset("data/facedata/facedatatrain",      "data/facedata/facedatatrainlabels");
        val   = loadDataset("data/facedata/facedatavalidation", "data/facedata/facedatavalidationlabels");
        test  = loadDataset("data/facedata/facedatatest",       "data/facedata/facedatatestlabels");
        return true;
    }
    return false;
}