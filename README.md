# Wildfire Detection Using Deep Learning Models

This repository contains Python implementations for wildfire detection using advanced deep learning models, including Swin V2, DenseNet, GANs, and committee-based approaches. The repository also includes experiments for hyperparameter tuning and feature variance testing.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Directory Structure](#directory-structure)
- [Experiments](#experiments)
- [Usage Instructions](#usage-instructions)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

To run the codes, you need to install the required libraries. Use the following commands to install them:

```bash
pip install torch torchvision
pip install transformers
pip install matplotlib
pip install scikit-learn
pip install numpy
```

## Dataset

The FLAME Dataset is required to run these experiments. Please download the training and test datasets from the following links:

- [Training Data](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs#)
- [Test Data](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones)
```bash
Code Folder/
├── FLAME_Dataset/
│   ├── Training/
│   └── Test/
```
## Directory Structure
The repository structure should be as follows:
```bash
Code Folder/
├── FLAME_Dataset/
│   ├── Training/
│   └── Test/
├── SwinV2-GAN.py
├── SwinPlus_DenseNetEnsemble.py
├── feature_variance.py
├── hyper_parameter.py
├── train_inception.py
├── evaluate_inception.py
├── committee_swin_wildfire_latest-3.py
└── voting_ensemble.py
```
## Experiments
Below are the available experiments in this repository:

### 1. Swin V2 + GAN
Run the script `SwinV2-GAN.py` to experiment with the Swin V2 model and GAN-based approach.

### 2. Swin V2 + DenseNet
Run the script `SwinPlus_DenseNetEnsemble.py` for the Swin V2 and DenseNet ensemble model.

### 3. Feature Variance Testing
Run `feature_variance.py` to analyze feature variance.

### 4. Hyperparameter Tuning
Run `hyper_parameter.py` to check the hyperparameter tuning code.

### 5. Inception V3
- Train the Inception V3 model using `train_inception.py`.
- Evaluate the trained model using `evaluate_inception.py`.

### 6. Committee-Based Approaches
- **Sampling-Based Approach**: Run `committee_swin_wildfire_latest-3.py`.
- **Model-Based Committee**: Run `voting_ensemble.py`.

## Usage Instructions

1. Open your terminal or command line interface.

2. Run the desired experiment using the following command:

   ```bash
   python <experiment_file_name>.py
```
Example
```bash
python SwinV2-GAN.py

```

### Note:
Depending on your operating system and Python version, you might need to use python3 instead of python.

## Contributing
If you’d like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Contributions are always welcome!

## License

This project is licensed under the MIT License.
