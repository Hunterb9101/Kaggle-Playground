# Kaggle Playground

## Problem Statement
Model development time for a Kaggle Playground competition is extremely limited. Thus it is in our best interests to follow a design pattern when trying to tackle these problems.

## Key Model Stages
- EDA
    - Check for nulls
    - Outlier Counts

- Data Processing
    - Feature Engineering
        - Standardization (For Linear Regression)
        - Feature Addition
        - Outlier Removal
        - Fill NAs
        - Categorical Encoding
        - PCA (Hyperparameter dependent)
        - Binning
    - Feature Selection
        - VIF (Linear Regression)
        - Correlation / Variation Pruning
        - Recursive Feature Elimination (Hyperparameter dependent)

- Model
    - Model Hyperparameter Optimization
    - Model training
    - Model Analytics
        - Metric scores
        - Top features

- Postprocessing
    - Round Final Inputs
    - Model Stack analytics
        - Metric Scores
        - Model Correlations
    - Model Stacking

## Key Notes
    - EDA: No output
