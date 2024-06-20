# NNBreastCancer

This repository contains the code for a breast cancer detection project using neural networks
## Introduction

Breast cancer is one of the most common types of cancer among women worldwide. Early detection and diagnosis are crucial for effective treatment and improved survival rates. This project aims to develop a neural network model to assist in the detection of breast cancer using histopathological images.

## Introduction

Breast cancer is one of the most common types of cancer among women worldwide. Early detection and diagnosis are crucial for effective treatment and improved survival rates. This project aims to develop a neural network model to assist in the detection of breast cancer using features from the sklearn breast cancer dataset.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset available in the `sklearn.datasets` module. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. It includes the following attributes:

- 30 real-valued features for each cell nucleus
- Target labels (0 = malignant, 1 = benign)

The dataset is loaded using the following code:

from sklearn.datasets import load_breast_cancer
breast_cancer_dataset = load_breast_cancer()

## Model Architecture

The neural network model used in this project is a simple feedforward neural network with the following architecture:

- Input layer: 30 features
- Hidden layers: 1 hidden layers with 20 
- Output layer: 1 neuron with sigmoid activation function

The model is implemented using TensorFlow and Keras.

## Installation

To run this project, you'll need to have Python and the following libraries installed:

- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install the required libraries using `pip`:

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib
