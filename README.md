# Does Representational Fairness Imply Empirical Fairness?

Source codes for AACL 2022 paper "Does Representational Fairness Imply Empirical Fairness?" 

If you use the code, please cite the following paper.

# Quick Links
+ [Overview](#overview)

+ [Requirements](#requirements)

+ [Data Preparation](#data-preparation)

+ [Source Code](#source-code)

+ [Experiments](#experiments)

# Overview

This repo contains the definition/implementation of contrastive learning in the fairness setting.

To replicate the experimental results, please checkout our fairlib https://github.com/HanXudong/fairlib.

# Requirements

The model is implemented using PyTorch and FairLib.

```
tqdm==4.62.3
numpy==1.22
docopt==0.6.2
pandas==1.3.4
scikit-learn==1.0
torch==1.10.0
PyYAML==6.0
seaborn==0.11.2
matplotlib==3.5.0
pickle5==0.0.12
transformers==4.11.3
sacremoses==0.0.53
```

Alternatively, you can install the fairlib directly:
```
pip install fairlib
```

# Data Preparation

```python
from fairlib import datasets

datasets.prepare_dataset("moji", "data/deepmoji")
datasets.prepare_dataset("bios", "data/bios")

```

# Source Code

## Contrastive Learning

`src/contrastive_loss` contains the implementation of contrastive learning in the fairness setting.

## Leakage

The leakage is estimated by a MLP discriminator, which is trained to predict protected attributes from hidden representations.
`src\leakage\leakage_estimation_bios.ipynb` shows examples of estimating leakages over bios dataset.

# Experiments

- **Hyperparameters:**  
    ```bash
    python main.py --FCL
    ```

    | Name                    | Default value | Description                                                     |
    |-------------------------|---------------|-----------------------------------------------------------------|
    | fcl_lambda_y            | 0.1           | strength of the supervised contrastive loss                     |
    | fcl_lambda_g            | 0.1           | strength of the fair supervised contrastive loss                |
    | fcl_temperature_y       | 0.01          | temperature for the fcl wrt main task learning                  |
    | fcl_temperature_g       | 0.01          | temperature for the fcl wrt protected attribute unlearning      |
    | fcl_base_temperature_y  | 0.01          | base temperature for the fcl wrt main task learning             |
    | fcl_base_temperature_g  | 0.01          | base temperature for the fcl wrt protected attribute unlearning |

- **Search Space:**  
    - same valued `fcl_lambda_y` and `fcl_lambda_g`: log-uniformly between 10^-3 ~ 10^1, 40 trials.

To replicate the full experimental results, please checkout fairlib https://github.com/HanXudong/fairlib.
