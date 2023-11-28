# Codon-Based-Organism-Discrimination
In this comprehensive project, I tackled the challenge of classifying codon usage across different biological kingdoms using a machine learning approach. 
The data was sourced from a codon usage dataset and comprised of various features related to codon frequencies in different species.

## Problem Addressed:
The primary objective was to develop a predictive model capable of classifying species into their respective biological kingdoms based on their codon usage patterns. 
This task involved handling imbalanced data, encoding categorical variables, and scaling features for optimal neural network performance.

## Methods Employed:

Preprocessed the data by one-hot encoding categorical variables, scaling numerical features using MinMaxScaler, and addressing class imbalances with SMOTE.
Split the data into training and test sets, converting them to NumPy arrays for compatibility with TensorFlow operations.
Designed and trained multiple deep neural networks with varying complexities, starting from a simple multilayer perceptron (MLP) to more sophisticated architectures involving principal component analysis (PCA) and sparsity learning.
Evaluated models using accuracy metrics and confusion matrices to determine the best performing architecture.

## Results:
The best-performing model, MLP_long, was composed of multiple dense layers, including hidden layers from previously trained models (H1 from mlp_high and HK from aec_test) and an additional dense layer (G). 
This ensemble approach leveraged the strengths of individual models and the principle of sparsity learning, leading to an outstanding accuracy rate of approximately 99%. 
The confusion matrix analysis confirmed the model's robustness, demonstrating high true positive rates across all biological kingdoms.
