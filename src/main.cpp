#include <iostream>
#include "DataLoader.h"
using namespace std;

int main() {
    Dataset train = loadDataset("data/digitdata/trainingimages",
                                "data/digitdata/traininglabels");

    cout << "Images: " << train.images.size() << "\n";
    cout << "Size: " << train.height << " x " << train.width << "\n";
    cout << "Classes: " << train.numClasses << "\n\n";

    for (const auto& row : train.images[0].rows) cout << row << "\n";
    cout << "Label: " << train.labels[0] << "\n";
    return 0;
}