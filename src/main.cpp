#include <iostream>
#include <string>
#include "DataLoader.h"
#include "Features.h"
#include "NaiveBayes.h"
#include "Perceptron.h"
#include "Evaluation.h"
#include "Demo.h"
using namespace std;

// Usage:   test.exe [digits|faces] [demo]
//   test.exe faces        -> run the experiments on faces
//   test.exe digits demo  -> interactive demo on digits
int main(int argc, char* argv[]) {
    string dataName = "digits";
    bool demo = false;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "demo") demo = true;
        else               dataName = arg;
    }

    Dataset train, val, test;
    if (!loadDatasets(dataName, train, val, test)) {
        cout << "Unknown dataset '" << dataName << "'. Use digits or faces.\n";
        return 1;
    }
    int numClasses = train.numClasses;
    cout << "Dataset: " << dataName << " (" << train.images.size() << " train, "
         << val.images.size() << " validation, " << test.images.size() << " test, "
         << train.height << "x" << train.width << " images)\n";

    if (demo) {
        runDemo(train, test);
        return 0;
    }

    int trials = 5;
    for (int useDensity = 0; useDensity <= 1; ++useDensity) {
        string featureName = useDensity ? "density" : "pixel";
        vector<FeatureVector> Xtrain = extractAll(train, useDensity);
        vector<FeatureVector> Xval   = extractAll(val,   useDensity);
        vector<FeatureVector> Xtest  = extractAll(test,  useDensity);

        NaiveBayes nb(1.0);
        Perceptron perceptron(10, 1, false);

        // Sanity check on VALIDATION data (not reported in the final results)
        nb.train(Xtrain, train.labels, numClasses);
        perceptron.train(Xtrain, train.labels, numClasses);
        cout << "[" << featureName << "] validation sanity check: NB "
             << 100 * accuracy(nb, Xval, val.labels) << "%, Perceptron "
             << 100 * accuracy(perceptron, Xval, val.labels) << "%\n";

        // The real experiments, on TEST data
        auto nbResults = runExperiment(nb, Xtrain, train.labels, Xtest, test.labels, numClasses, trials);
        printResults(dataName + ": Naive Bayes / " + featureName, nbResults);
        writeCsv(dataName + "_nb_" + featureName + ".csv", nbResults);

        auto pResults = runExperiment(perceptron, Xtrain, train.labels, Xtest, test.labels, numClasses, trials);
        printResults(dataName + ": Perceptron / " + featureName, pResults);
        writeCsv(dataName + "_perceptron_" + featureName + ".csv", pResults);
    }
    return 0;
}