#include "Demo.h"
#include <iostream>
#include "Features.h"
#include "NaiveBayes.h"
#include "Perceptron.h"
using namespace std;

void runDemo(const Dataset& train, const Dataset& test) {
    // ---- 1. let the user choose the algorithm and the features ----
    int algo = 0, feat = 0;
    cout << "Algorithm  (1 = Naive Bayes, 2 = Perceptron): ";
    cin >> algo;
    cout << "Features   (1 = pixel, 2 = density): ";
    cin >> feat;
    bool useDensity = (feat == 2);

    // ---- 2. extract features and train on 100% of the training data ----
    vector<FeatureVector> Xtrain = extractAll(train, useDensity);

    NaiveBayes nb(1.0);
    Perceptron perceptron(10, 1, false);
    Classifier* model;                       // points at whichever one we chose
    if (algo == 1) model = &nb;
    else           model = &perceptron;

    cout << "Training...\n";
    model->train(Xtrain, train.labels, train.numClasses);
    cout << "Done.\n";

    // ---- 3. look up test images one at a time ----
    while (true) {
        cout << "\nTest image number (0 to " << test.images.size() - 1 << ", -1 to quit): ";
        int index;
        if (!(cin >> index) || index < 0) break;
        if (index >= (int)test.images.size()) {
            cout << "Out of range.\n";
            continue;
        }

        const Image& img = test.images[index];
        for (const string& row : img.rows) cout << row << "\n";

        FeatureVector features = extractOne(img, useDensity);   // feature extraction
        int predicted = model->predict(features);               // prediction
        cout << "Number of features: " << features.size() << "\n";
        cout << "Predicted label: " << predicted << "\n";
        cout << "True label:      " << test.labels[index] << "\n";
        cout << (predicted == test.labels[index] ? "CORRECT\n" : "WRONG\n");
    }
}