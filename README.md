# Machine Learning System for Anomaly Detection in the MNIST Dataset

This project describes a machine learning system designed for anomaly detection in the MNIST dataset, utilizing an approach that combines semi-supervised learning and active learning. The main neural network, a **Multi-Layer Perceptron (MLP)**, is pre-trained with an **autoencoder** to obtain a compact and effective representation of the data. The model uses the hypersphere technique, known as **Support Vector Data Description (SVDD)**, to identify anomalies within the feature space.

The **active labeling algorithm** optimizes the selection of the most informative data samples, improving the effectiveness of training with a limited labeling budget. A detailed description of the learning paradigm, network structure, training process, and labeling algorithm is provided, highlighting the importance of pre-training and budget management. This integrated approach is particularly useful for identifying anomalies in real-world contexts where computational resources and labeled data are limited.

## Introduction

This project focuses on developing a machine learning system for anomaly detection in the MNIST dataset. By combining **semi-supervised learning** and **active learning**, the system efficiently identifies anomalies with limited labeled data and computational resources.

The main neural network, a **Multi-Layer Perceptron (MLP)**, is pre-trained using an **autoencoder** to achieve a compact and effective representation of the data. The model employs the **Support Vector Data Description (SVDD)** technique to detect anomalies within the feature space.

The **active labeling algorithm** optimizes the selection of the most informative data samples to label, enhancing training effectiveness within a limited labeling budget. This approach is particularly beneficial in real-world scenarios where labeled data is scarce and expensive to obtain.

## Features

- **Semi-Supervised Learning**: Leverages unlabeled data to improve model performance.
- **Active Learning**: Intelligently selects the most informative samples to label, optimizing the use of the labeling budget.
- **Pre-Training with Autoencoder**: Reduces dimensionality and extracts significant features from the data.
- **Support Vector Data Description (SVDD)**: Detects anomalies by defining a hypersphere in the feature space that encloses normal data.
- **Application to MNIST Dataset**: Validates the approach on a standard handwritten digit recognition dataset.

## Model Architecture

- **Autoencoder**:
  - An unsupervised neural network used for pre-training.
  - Composed of an encoder and decoder to compress and reconstruct data.
- **Multi-Layer Perceptron (MLP)**:
  - A feedforward neural network used as the main classifier.
  - Trained using the representations learned from the autoencoder.
- **SVDD Algorithm**:
  - An anomaly detection technique based on defining a hypersphere that encloses normal data in the feature space.
- **Active Labeling Algorithm**:
  - Iteratively selects the most informative samples to label.
  - Improves training efficiency with a limited labeling budget.

## Dataset

The dataset used is the **MNIST** (Modified National Institute of Standards and Technology) dataset, consisting of 28x28 pixel images of handwritten digits.

- **Number of Classes**: 10 (digits from 0 to 9)
- **Dataset Size**: 70,000 images
- **Data Format**: Grayscale images

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- PyTorch and torchvision libraries
- Other dependencies listed in `requirements.txt`
