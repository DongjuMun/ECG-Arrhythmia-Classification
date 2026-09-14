# ECG Arrhythmia Classification

This project implements a machine-learning pipeline for classifying ECG heartbeats using features extracted from the **MIT-BIH Arrhythmia Database**.

The original dataset contains five heartbeat categories:

- **N** — Normal
- **SVEB** — Supraventricular Ectopic Beat
- **VEB** — Ventricular Ectopic Beat
- **F** — Fusion Beat
- **Q** — Unknown Beat

For the final modeling task, the **Unknown Beat (Q)** category was removed because it contained only 15 observations, making reliable training and evaluation difficult. Therefore, the final classification problem consists of four classes: **N, SVEB, VEB, and F**.

## Project Overview

The project includes the following steps:

- Dataset inspection and exploratory data analysis (EDA)
- Removal of duplicated and redundant ECG features
- Record-level train, validation, and test split
- Feature standardization
- Principal Component Analysis (PCA)
- Multinomial logistic regression implemented from scratch using NumPy
- Random Oversampling with XGBoost
- SMOTE with XGBoost
- Regularization experiments
- Model evaluation using accuracy, precision, recall, F1-score, macro-F1, and confusion matrices

The dataset is split at the **ECG-record level** instead of randomly splitting individual heartbeat observations. This prevents heartbeats belonging to the same ECG recording from appearing in both the training and evaluation subsets.

## Dataset

The project uses the **ECG Arrhythmia Classification Dataset**, which contains ECG-derived features extracted from the MIT-BIH Arrhythmia Database.

The CSV file expected by the program is:

```text
MIT-BIH Arrhythmia Database.csv
```

Place the dataset in the same directory as the Python script, or modify the dataset path in the Python file:

```python
csv_path = Path(__file__).resolve().parent / "MIT-BIH Arrhythmia Database.csv"
```

## Requirements

The project requires **Python 3** and the following libraries:

- NumPy
- pandas
- Matplotlib
- scikit-learn
- imbalanced-learn
- XGBoost

The required libraries can be installed using:

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost
```

## Running the Program

Place the Python script and CSV dataset in the same directory.

Example:

```text
project/
│
├── a01712119_dongjumun_ecg.py
├── MIT-BIH Arrhythmia Database.csv
└── README.md
```

Run the program with:

```bash
python a01712119_dongjumun_ecg.py
```

Depending on the system, you may need to use:

```bash
python3 a01712119_dongjumun_ecg.py
```

## Models

### 1. Multinomial Logistic Regression

The baseline model is a **multinomial logistic regression classifier implemented from scratch using NumPy**, without using a machine-learning framework for model training.

The implementation includes:

- Softmax activation
- Categorical cross-entropy loss
- Batch gradient descent
- Manual calculation of precision, recall, F1-score, and confusion matrices

Before logistic regression, PCA is applied to the standardized features. The minimum number of principal components required to explain at least **95% of the variance** is retained.

### 2. XGBoost with Random Oversampling

Random Oversampling is applied exclusively to the **training dataset** to increase the representation of minority heartbeat classes.

The resulting balanced training dataset is then used to train an XGBoost multiclass classifier.

### 3. XGBoost with SMOTE

SMOTE (**Synthetic Minority Over-sampling Technique**) is also applied exclusively to the training dataset.

Unlike Random Oversampling, which duplicates existing observations, SMOTE generates synthetic minority-class observations using neighboring samples in the feature space.

The **SMOTE-XGBoost** configuration produced the highest validation macro-F1 among the evaluated models and was therefore selected as the final model.

## Model Evaluation

The ECG dataset is highly imbalanced, with the Normal (N) class representing the majority of observations.

For this reason, **accuracy alone is not sufficient** for evaluating model performance.

The following metrics are considered:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Confusion matrix

**Macro-F1** is used as the primary model-selection metric because it gives equal importance to each heartbeat class regardless of its frequency in the dataset.

## Reproducibility

A random seed of **42** is used for:

- Record-level data splitting
- Random Oversampling
- SMOTE
- XGBoost

This improves the reproducibility of the experiments.

However, small numerical differences may occur when executing the program in different environments, operating systems, hardware configurations, or library versions. For example, results obtained using **Google Colab** may differ slightly from results obtained on a local computer.

These small differences do not necessarily change the overall conclusions of the experiment.

## Main Results

The baseline multinomial logistic regression achieved high overall accuracy but relatively low macro-F1 due to its limited ability to classify minority heartbeat classes.

The final **SMOTE-XGBoost** model improved balanced multiclass performance compared with the baseline model, particularly for **VEB** and partially for **SVEB**.

However, the model continued to have difficulty detecting the **Fusion Beat (F)** class, demonstrating that severe class imbalance remains an important limitation.

## Author

**Dongju Mun**  
A01712119  
Tecnológico de Monterrey
