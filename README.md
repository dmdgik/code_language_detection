# Code language detector

## Overview

Neural network that allows you to determine the programming language in which the text is written.

A list of defined languages is available [here](https://github.com/dmdgik/code_language_detection/blob/main/configs/languages.yaml).

Two neural networks are presented that allow you to determine the programming language - based on the LSTM architecture and based on the BERT architecture.

## Common

Create conda virtual env:

```console
conda create -n code_lang python=3.9
```

```console
conda activate code_lang
```

Install requirements

```console
pip install -r requirements.txt
```

You will need to create a secrets.yaml file at configs folder to save the HUGGINGFACE_AUTH_TOKEN, AWS_S3_ACCESS_KEY_ID, AWS_S3_SECRET_ACCESS_KEY.

[CodeBERT](https://huggingface.co/microsoft/codebert-base) used as tokenizer for LSTM and BERT-based classifier as well as base model for BERT-based model

Loading tokenizer and base model by this [config](https://github.com/dmdgik/code_language_detection/blob/main/configs/common.yaml)

```console
cd src/CodeLanguageDetector/running_scripts
python common.py
```

## Data

500k samples per train 

140k samples per validation

70k samples per test

max length 512 tokens

[Bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) data used for collecting examples of code texts

Result [datasets](https://www.kaggle.com/datasets/dmdgik/code-language-data). You can just download it.

The data is divided into train, validation and test. Also, the data is presented in two forms - full and partial. That is, it is possible to use both full texts with codes and parts of these texts for training.

Scripts for getting data by this data [config](https://github.com/dmdgik/code_language_detection/blob/main/configs/data_config.yaml):

Raw data:

```console
cd src/CodeLanguageDetector/running_scripts
python get_raw_data.py
```

Processed data: 

```
cd src/CodeLanguageDetector/running_scripts
python get_processed_data.py
```

## Models

[File](https://github.com/dmdgik/code_language_detection/blob/main/src/CodeLanguageDetector/models/models.py) with models code

## Datasets

[File](https://github.com/dmdgik/code_language_detection/blob/main/src/CodeLanguageDetector/models/datasets.py) with dataset code

## Training scripts

BERT model training [script](https://github.com/dmdgik/code_language_detection/blob/main/src/CodeLanguageDetector/running_scripts/train_bert.py)

BERT model training [config](https://github.com/dmdgik/code_language_detection/blob/main/configs/experiment_bert.yaml)

LSTM model training [script](https://github.com/dmdgik/code_language_detection/blob/main/src/CodeLanguageDetector/running_scripts/train_lstm.py)

LSTM model training [config](https://github.com/dmdgik/code_language_detection/blob/main/configs/experiment_lstm.yaml)

Commands for running (configure your accelerate before)

```console
cd src/CodeLanguageDetector/running_scripts
accelerate launch train_bert.py --experiment_config="../../../configs/experiment_bert.yaml"
```

or

```console
cd src/CodeLanguageDetector/running_scripts
accelerate launch train_lstm.py --experiment_config="../../../configs/experiment_lstm.yaml"
```

## Training kaggle notes

You can train models with kaggle notebooks, just upload them, fill token values and run

BERT model training [notebook](https://github.com/dmdgik/code_language_detection/blob/main/notebooks/kaggle-code-language-detector-bert.ipynb)

BERT model training [config](https://github.com/dmdgik/code_language_detection/blob/main/configs/kaggle_experiment_bert.yaml)

LSTM model training [notebook](https://github.com/dmdgik/code_language_detection/blob/main/notebooks/kaggle-code-language-detector-lstm.ipynb)

LSTM model training [config](https://github.com/dmdgik/code_language_detection/blob/main/configs/kaggle_experiment_lstm.yaml)

## Results

BERT-based [model](https://github.com/dmdgik/code_language_detection/blob/main/models/obtained/checkpoint_bert_model_training_unfreezed_bert.pt) full text test accuracy: 97.4%

LSTM-based [model](https://github.com/dmdgik/code_language_detection/blob/main/models/obtained/checkpoint_lstm_model_training_initial.pt) full text test accuracy: 96.4%

Training process freezed [results](https://github.com/dmdgik/code_language_detection/blob/main/models/obtained/epochs_results_bert_model_training_freezed_bert.yaml) and final [funetuning](https://github.com/dmdgik/code_language_detection/blob/main/models/obtained/epochs_results_bert_model_training_unfreezed_bert.yaml) BERT

Training process [results](https://github.com/dmdgik/code_language_detection/blob/main/models/obtained/epochs_results_lstm_model_training_initial.yaml) LSTM
