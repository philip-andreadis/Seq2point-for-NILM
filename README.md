# Seq2point-for-NILM

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.7.0-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Description

Implementation of the sequence-to-point CNN methodology for Non-Intrusive Load Monitoring as part of a research internship at Powerchainger Groningen.

The seq2point model has been trained and tested on different sampling frequencies of two datasets, namely REDD and UK-DALE. 
Cross-domain testing between the two datasets with an optional fine-tuning step is also available. 


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Install dependencies in requirements file. 


```bash
pip install -r requirements.txt
```

## Usage

All the experiments can be run by executing the following script:

python ./main_exp.py --train_houses 2 5 --test_houses 1 -sd redd -td redd --device refrigerator -lr 0.001 -e 20 -b 64

### Arguments

| Argument                | Type      | Default     | Description                                     |
|-------------------------|-----------|-------------|-------------------------------------------------|
| --train_houses          | int list  | (Required)  | List of houses for training                     |
| --test_houses           | int list  | (Required)  | List of houses for testing                      |
| --device                | str       | refrigerator| Target device to train on                        |
| --nas                   | str       |             | How to deal with missing values. Possible values: 'interpolate', 'drop' |
| -sr, --sampling_rate    | int       | 1           | Target sampling rate                            |
| -st, --standardise      | bool      | False       | Standardize data                                |
| -lr, --learning_rate    | float     | 0.001       | Learning rate for the model                      |
| -e, --epochs            | int       | 20          | Number of training epochs                       |
| -b, --batch             | int       | 32          | Batch size                                      |
| --loss                  | str       | mse         | Loss function to use for training               |
| -sd, --source_domain    | str       | redd        | Which source domain will be used. Options are 'redd' or 'ukdale' |
| -td, --target_domain    | str       | redd        | Which target domain will be used. Options are 'redd' or 'ukdale' |
| -tl, --transfer_learning | bool      | False       | Enable fine tuning                        |

 
