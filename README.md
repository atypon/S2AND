# S2AND
This repository provides access to the S2AND dataset and S2AND reference model described in the paper [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421) by Shivashankar Subramanian, Daniel King, Doug Downey, Sergey Feldman.

The reference model is live on semanticscholar.org, and the trained model is available now as part of the data download (see below).

## Installation
To install this package, run the following:

```bash
git clone https://github.com/atypon/S2AND.git
cd S2AND
conda create -y --name s2and python==3.7
conda activate s2and
pip install -r requirements.in
```

If you run into cryptic errors about GCC on macOS while installing the requirments, try this instead:
```bash
CFLAGS='-stdlib=libc++' pip install -r requirements.in
```

## Data 
To obtain the S2AND dataset, run the following command after the package is installed (from inside the `S2AND` directory):  

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release data/`

Note that this software package comes with tools specifically designed to access and model the dataset.

For the data extended with PKG's info space, run the following command :

`gsutil -m cp -r \
  "gs://pkg-datasets/datasets/S2AND/aminer/" \
  "gs://pkg-datasets/datasets/S2AND/arnetminer/" \
  "gs://pkg-datasets/datasets/S2AND/kisti/" \
  "gs://pkg-datasets/datasets/S2AND/pubmed/" \
  "gs://pkg-datasets/datasets/S2AND/zbmath/" \
  extended_data/`

## Configuration
Modify the config file at `data/path_config.json`. This file should look like this
```
{
    "main_data_dir": "absolute path to wherever you downloaded the data to",
    "internal_data_dir": "ignore this one unless you work at AI2"
}
```
As the dummy file says, `main_data_dir` should be set to the location of wherever you downloaded the data to, and
`internal_data_dir` can be ignored, as it is used for some scripts that rely on unreleased data, internal to Semantic Scholar.

## Run

There are three main run scripts to perform the disambiguation process.

  * run_inference.py: Produces embeddings with the selected transformer model defined by an onnx file for all signatures in the data.
  * run_lightgbm_training.py: Trains lightgbm classifier to calculate distance between signatures.
  * run_clustering.py: Performs hyperparam optimization of the clusterer and clusters the signatures.
