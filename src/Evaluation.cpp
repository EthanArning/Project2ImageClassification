#include "Evaluation.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
using namespace std;

double accuracy(const Classifier& model, const vector<FeatureVector>& X, const vector<int>& y) {
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i)
        if (model.predict(X[i]) == y[i]) correct++;
    return static_cast<double>(correct) / X.size();
}

// Mean and standard deviation of a list of numbers.
static void meanAndStd(const vector<double>& v, double& mean, double& sd) {
    mean = accumulate(v.begin(), v.end(), 0.0) / v.size();
    double sumSq = 0.0;
    for (double x : v) sumSq += (x - mean) * (x - mean);
    sd = sqrt(sumSq / v.size());
}

vector<PercentResult> runExperiment(Classifier& model,
                                    const vector<FeatureVector>& Xtrain, const vector<int>& ytrain,
                                    const vector<FeatureVector>& Xtest,  const vector<int>& ytest,
                                    int numClasses, int trials) {
    mt19937 rng(42);
    vector<PercentResult> results;

    for (int percent = 10; percent <= 100; percent += 10) {
        size_t n = Xtrain.size() * percent / 100;     // how many training images to use
        vector<double> accs, times;

        for (int t = 0; t < trials; ++t) {
            // ---- pick n random training images ----
            vector<int> order(Xtrain.size());
            iota(order.begin(), order.end(), 0);
            shuffle(order.begin(), order.end(), rng);

            vector<FeatureVector> Xsub;
            vector<int> ysub;
            for (size_t i = 0; i < n; ++i) {
                Xsub.push_back(Xtrain[order[i]]);
                ysub.push_back(ytrain[order[i]]);
            }

            // ---- train, timing ONLY the training ----
            auto start = chrono::steady_clock::now();
            model.train(Xsub, ysub, numClasses);
            auto stop = chrono::steady_clock::now();
            times.push_back(chrono::duration<double, milli>(stop - start).count());

            // ---- test on the TEST data ----
            accs.push_back(accuracy(model, Xtest, ytest));
        }

        PercentResult r;
        r.percent = percent;
        meanAndStd(accs, r.meanAcc, r.stdAcc);
        meanAndStd(times, r.meanTimeMs, r.stdTimeMs);
        results.push_back(r);
    }
    return results;
}

void printResults(const string& title, const vector<PercentResult>& results) {
    cout << "\n=== " << title << " ===\n";
    cout << setw(7) << "Train%" << setw(11) << "MeanAcc%" << setw(10) << "StdAcc%"
         << setw(14) << "MeanTime(ms)" << setw(13) << "StdTime(ms)" << "\n";
    cout << fixed << setprecision(2);
    for (const PercentResult& r : results)
        cout << setw(7) << r.percent << setw(11) << 100 * r.meanAcc << setw(10) << 100 * r.stdAcc
             << setw(14) << r.meanTimeMs << setw(13) << r.stdTimeMs << "\n";
}

void writeCsv(const string& path, const vector<PercentResult>& results) {
    ofstream out(path);
    out << "percent,mean_accuracy,std_accuracy,mean_train_ms,std_train_ms\n";
    for (const PercentResult& r : results)
        out << r.percent << "," << r.meanAcc << "," << r.stdAcc << ","
            << r.meanTimeMs << "," << r.stdTimeMs << "\n";
}