#include <iostream>
#include "DataLoader.h"
#include "Features.h"
#include "NaiveBayes.h"
#include "Perceptron.h"
#include "Evaluation.h"
using namespace std;

// Turn every image in a dataset into a feature vector.
vector<FeatureVector> extractAll(const Dataset& data, bool useDensity) {
    vector<FeatureVector> all;
    for (const Image& img : data.images) {
        if (useDensity) all.push_back(extractDensity(img, 4, 4, 3));
        else            all.push_back(extractPixel(img));
    }
    return all;
}

int main() {
    Dataset train = loadDataset("data/digitdata/trainingimages",   "data/digitdata/traininglabels");
    Dataset val   = loadDataset("data/digitdata/validationimages", "data/digitdata/validationlabels");
    Dataset test  = loadDataset("data/digitdata/testimages",       "data/digitdata/testlabels");
    int numClasses = train.numClasses;
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
        printResults("Naive Bayes / " + featureName, nbResults);
        writeCsv("digits_nb_" + featureName + ".csv", nbResults);

        auto pResults = runExperiment(perceptron, Xtrain, train.labels, Xtest, test.labels, numClasses, trials);
        printResults("Perceptron / " + featureName, pResults);
        writeCsv("digits_perceptron_" + featureName + ".csv", pResults);
    }
    return 0;
}