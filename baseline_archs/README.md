# Running Revdict

The following sections walk through the process for setting up the environment, running training, prediction, and 
scoring.

## Disclaimer

This codebase is based on the baseline code provided at https://github.com/TimotheeMickus/codwoe. The novel 
contributions of our approach are found in codwoe/baseline_archs/code/data_ngram.py which implements dataset generation 
according to the minimal, ascii, and comprehensive n-gram tokenization schemes described in the project report and 
presentation.

We have also added codwoe/analysis/dimensionality_reduction.py for testing visualizations, as well as 
codwoe/docker/setup.sh and everything contained in the codwoe/scripts directory to aid in setting up the environment.

## Dataset:
* Go to https://codwoe.atilf.fr/ 
* Download "Train and development datasets". Exact "en.train.json" and "en.dev.json" from the download zip. 
* Download "Test datasets". Exact "en.test.defmod.json" from the download zip.
* Download "Reference data for scoring program". Exact "en.test.defmod.complete.json" from the download zip.
* Place all the exacted json files into the codwoe/data directory.

## Environment Setup

First, ensure the data is placed within the codwoe/data directory, as this is required for the environment setup.

In order to set up the environment, navigate to the codwoe/docker directory and run 
```shell
./setup.sh
```
This builds a container using the Dockerfile and copies all necessary code and data into the container.

## Running the Code

The scripts contained in the codwoe/scripts directory can be used to invoke the python commands needed for running 
training, prediction, and model scoring. The revdict scripts each accept two arguments:

1. The embedding type to use ("sgns", "char", or "electra")
2. The n-gram tokenization scheme to use ("min", "ascii", or "comp")

As an example, to run the process for minimal n-gram tokenization to predict "char" embeddings, you would run the 
following from inside the container:

```shell
./home/codwoe/scripts/train_revdict char min
./home/codwoe/scripts/predict_revdict char min
./home/codwoe/scripts/score_revdict char min
```

The scores will be generated at 
/home/codwoe/baseline_archs/model/revdict-<tokenization_scheme>-ngram/<embedding_type>/scores.txt,
where <tokenization_scheme> is one of: ["minimal", "ascii", "comp"], and <embedding_type> is one of: 
["sgns", "char", "electra"]. For the example above, the scores will be placed at:

```shell
/home/codwoe/baseline_archs/model/revdict-minimal-ngram/char/scores.txt
```
