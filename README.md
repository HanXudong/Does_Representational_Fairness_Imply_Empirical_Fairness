# Does Representational Fairness Imply Empirical Fairness?

Source codes for AACL 2022 paper "Does Representational Fairness Imply Empirical Fairness?" 

If you use the code, please cite the following paper.

```
@inproceedings{shen-etal-2022-representational,
    title = "Does Representational Fairness Imply Empirical Fairness?",
    author = "Shen, Aili  and
      Han, Xudong  and
      Cohn, Trevor  and
      Baldwin, Timothy  and
      Frermann, Lea",
    booktitle = "Findings of the Association for Computational Linguistics: AACL-IJCNLP 2022",
    month = nov,
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-aacl.8",
    pages = "81--95",
    abstract = "NLP technologies can cause unintended harms if learned representations encode sensitive attributes of the author, or predictions systematically vary in quality across groups. Popular debiasing approaches, like adversarial training, remove sensitive information from representations in order to reduce disparate performance, however the relation between representational fairness and empirical (performance) fairness has not been systematically studied. This paper fills this gap, and proposes a novel debiasing method building on contrastive learning to encourage a latent space that separates instances based on target label, while mixing instances that share protected attributes. Our results show the effectiveness of our new method and, more importantly, show across a set of diverse debiasing methods that \textit{representational fairness does not imply empirical fairness}. This work highlights the importance of aligning and understanding the relation of the optimization objective and final fairness target.",
}

```

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
