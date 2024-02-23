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


| Argument   | Description                      | Example Usage          |
|------------|----------------------------------|------------------------|
| `--arg1`   | Description of argument 1.       | `--arg1 value1`        |
| `--arg2`   | Description of argument 2.       | `--arg2 value2`        |
 
