#include <iostream>
#include "DataLoader.h"
#include "Features.h"
using namespace std;

int main() {
    Dataset train = loadDataset("data/digitdata/trainingimages",
                                "data/digitdata/traininglabels");

    const Image& img = train.images[0];
    for (const string& row : img.rows) cout << row << "\n";
    cout << "Label: " << train.labels[0] << "\n\n";

    FeatureVector pixel = extractPixel(img);
    cout << "Pixel features: " << pixel.size() << " numbers\n";
    int ones = 0;
    for (int f : pixel) ones += f;
    cout << "  of which " << ones << " are 1 (ink)\n\n";

    FeatureVector density = extractDensity(img, 4, 4, 3);
    cout << "Density features: " << density.size() << " numbers\n";
    cout << "  as a 7x7 grid:\n";
    for (size_t i = 0; i < density.size(); ++i) {
        cout << (density[i] ? '#' : '.');
        if ((i + 1) % 7 == 0) cout << "\n";
    }
    return 0;
}