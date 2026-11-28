#pragma once
#include "Types.h"

using namespace std;

Dataset loadDataset(const string& imagePath, const string& labelPath);

// Loads the train / validation / test sets for "digits" or "faces".
// Returns false if the name isn't recognised.
bool loadDatasets(const string& name, Dataset& train, Dataset& val, Dataset& test);