# -*- coding: utf-8 -*-
""" A01712119_DongjuMun_ECG.ipynb """

import numpy as np
import pandas as pd
from pathlib import Path

csv_path = Path(__file__).resolve().parent / "MIT-BIH Arrhythmia Database.csv"
df = pd.read_csv(csv_path)
print(df.head())

print(df.info())

print(df.describe())

duplicate_rows = df[df.duplicated()]
print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")

if not duplicate_rows.empty:
    print("\nFirst 5 duplicate rows:")
    print(duplicate_rows.head())

columns_0 = [col for col in df.columns if col.startswith('0_')]
columns_1 = [col for col in df.columns if col.startswith('1_')]

print("Comparing features with '0_' and '1_' prefixes:")
for col_0 in columns_0:
    # Extract the suffix
    suffix = col_0[2:]
    col_1 = '1_' + suffix

    if col_1 in columns_1:
        print(f"\n--- Comparing {col_0} and {col_1} ---")

        exact_matches = (df[col_0] == df[col_1]).sum()
        total_rows = len(df)
        percentage_match = (exact_matches / total_rows) * 100
        print(f"Percentage of exact matches: {percentage_match:.2f}%")

df['pre-RR'] = df['0_pre-RR']
df['post-RR'] = df['0_post-RR']
df = df.drop(columns=['0_pre-RR', '0_post-RR', '1_pre-RR', '1_post-RR'])
print(df.head())

import matplotlib.pyplot as plt

plt.hist(df['type'], bins=50)
plt.xlabel('type')
plt.ylabel('frequency')
plt.title('Histogram of type')
plt.show()

counts = df["type"].value_counts()
percentages = df["type"].value_counts(normalize=True) * 100

class_distribution = pd.DataFrame({
    "Count": counts,
    "Percentage": percentages
})

print(class_distribution)

df = df[df["type"] != "Q"]

counts = df["type"].value_counts()
percentages = df["type"].value_counts(normalize=True) * 100

class_distribution = pd.DataFrame({
    "Count": counts,
    "Percentage": percentages
})

print(class_distribution)

df.drop(columns=["type", "record"]).hist(
    figsize=(15, 15),
    bins=30
)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

numeric_df = df.drop(columns=["type", "record"])

corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(16, 14))

im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr.columns, fontsize=7)

fig.colorbar(im, ax=ax, label="Correlation")

ax.set_title("Correlation Matrix of ECG Features")

plt.tight_layout()
plt.show()

import numpy as np

corr = numeric_df.corr()

# Get only upper triangle so we don't get duplicate pairs
upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

# Convert to list of feature pairs
pairs = (
    upper.stack()
    .reset_index()
)

pairs.columns = ["Feature 1", "Feature 2", "Correlation"]

# Add absolute correlation for sorting
pairs["Absolute Correlation"] = pairs["Correlation"].abs()

# Sort strongest relationships first
pairs = pairs.sort_values(
    "Absolute Correlation",
    ascending=False
)

print(pairs.head(30))


df = df.drop(columns=['0_qrs_morph0', '1_qrs_morph0'])
print(df.head())

features = df.drop(columns=[
    "type",
    "record"
])
target = df["type"]

np.random.seed(42)

# Get unique ECG records
records = df["record"].unique()

# Shuffle record IDs
np.random.shuffle(records)

# 60% of records for training
split_index = int(len(records) * 0.6)

# 20% of records for validation/tests
split_index_2 = split_index + int(len(records) * 0.2)

# Split records into training, validation, and test sets
train_records = records[:split_index]
validation_records = records[split_index:split_index_2]
test_records = records[split_index_2:]

# Create masks
train_mask = df["record"].isin(train_records)
validation_mask = df["record"].isin(validation_records)
test_mask = df["record"].isin(test_records)

# Split using the features DataFrame to ensure only numeric data
X_train = features[train_mask].to_numpy()
X_val = features[validation_mask].to_numpy()
X_test = features[test_mask].to_numpy()

y_train = target[train_mask].to_numpy()
y_val = target[validation_mask].to_numpy()
y_test = target[test_mask].to_numpy()

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

print("Train %:", len(X_train) / len(features) * 100)
print("Validation %:", len(X_val) / len(features) * 100)
print("Test %:", len(X_test) / len(features) * 100)

mean_train = np.mean(X_train, axis=0)
std_train = np.std(X_train, axis=0)
std_train[std_train == 0] = 1
X_train_scaled = (X_train - mean_train) / std_train
X_val_scaled = (X_val - mean_train) / std_train
X_test_scaled = (X_test - mean_train) / std_train
print(np.mean(X_train_scaled, axis=0))
print(np.std(X_train_scaled, axis=0))

X = X_train_scaled
y = y_train

# Calculating the covariance matrix
covariance_matrix = np.cov(X.T)

# Using np.linalg.eigh function
eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
# Sort eigenvalues from largest to smallest
indices = np.argsort(eigen_values)[::-1]

eigen_values = eigen_values[indices]
eigen_vectors = eigen_vectors[:, indices]

# Calculating the explained variance on each of components
variance_explained = []
for i in eigen_values:
     variance_explained.append((i/sum(eigen_values))*100)

print(variance_explained)

# Identifying components that explain at least 95%
cumulative_variance_explained = np.cumsum(variance_explained)
print(cumulative_variance_explained)

n_components = np.argmax(cumulative_variance_explained >= 95) + 1

print(n_components)
print(cumulative_variance_explained[n_components - 1])

selected_eigenvectors = eigen_vectors[:, :n_components]

X_train_pca = X_train_scaled @ selected_eigenvectors
X_val_pca = X_val_scaled @ selected_eigenvectors
X_test_pca = X_test_scaled @ selected_eigenvectors

print("Train X: ", X_train_pca.shape)
print("Validation X: ", X_val_pca.shape)
print("Test X: ", X_test_pca.shape)

class_to_idx = {
    "N": 0,
    "SVEB": 1,
    "VEB": 2,
    "F": 3
}

idx_to_class = {
    0: "N",
    1: "SVEB",
    2: "VEB",
    3: "F"
}

y_train_num = np.array([class_to_idx[label] for label in y_train])
y_val_num = np.array([class_to_idx[label] for label in y_val])
y_test_num = np.array([class_to_idx[label] for label in y_test])

print("Train class distribution:", np.bincount(y_train_num))
print("Validation class distribution:", np.bincount(y_val_num))
print("Test class distribution:", np.bincount(y_test_num))

Y_train = np.eye(4)[y_train_num]
Y_val = np.eye(4)[y_val_num]
Y_test = np.eye(4)[y_test_num]
print(Y_train.shape)
print(Y_train[:5])

def softmax(Z):

    # Numerical stability
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)

    exp_Z = np.exp(Z_shifted)

    probabilities = exp_Z / np.sum(
        exp_Z,
        axis=1,
        keepdims=True
    )

    return probabilities

def cross_entropy_loss(Y, Y_hat):

    m = Y.shape[0]

    epsilon = 1e-15

    Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)

    loss = -np.sum(Y * np.log(Y_hat)) / m

    return loss

def train_logistic_regression(
    X,
    y,
    n_classes,
    learning_rate=0.01,
    epochs=1000
):

    m, n_features = X.shape

    # One-hot encoding
    Y = np.eye(n_classes)[y]

    # Initialize parameters
    W = np.zeros((n_features, n_classes))
    b = np.zeros((1, n_classes))

    loss_history = []

    for epoch in range(epochs):

        # Forward propagation
        Z = X @ W + b

        Y_hat = softmax(Z)

        # Loss
        loss = cross_entropy_loss(
            Y,
            Y_hat
        )

        loss_history.append(loss)

        # Backpropagation
        error = Y_hat - Y

        dW = (X.T @ error) / m

        db = np.sum(
            error,
            axis=0,
            keepdims=True
        ) / m

        # Gradient descent
        W -= learning_rate * dW
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}: "
                f"Loss = {loss:.4f}"
            )

    return W, b, loss_history

W, b, loss_history = train_logistic_regression(
    X_train_pca,
    y_train_num,
    n_classes=4,
    learning_rate=0.01,
    epochs=1000
)

def predict(X, W, b):

    Z = X @ W + b

    probabilities = softmax(Z)

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return predictions

y_train_pred = predict(
    X_train_pca,
    W,
    b
)

y_val_pred = predict(
    X_val_pca,
    W,
    b
)

train_accuracy = np.mean(
    y_train_pred == y_train_num
)

val_accuracy = np.mean(
    y_val_pred == y_val_num
)

def evaluate_model(y_true, y_pred, class_names):

    num_classes = len(class_names)

    # Confusion matrix
    conf_matrix = np.zeros(
        (num_classes, num_classes),
        dtype=int
    )

    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):

        TP = conf_matrix[i, i]

        FP = np.sum(conf_matrix[:, i]) - TP
        FN = np.sum(conf_matrix[i, :]) - TP

        precision = (
            TP / (TP + FP)
            if (TP + FP) > 0
            else 0
        )

        recall = (
            TP / (TP + FN)
            if (TP + FN) > 0
            else 0
        )

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return (
        accuracy,
        conf_matrix,
        precisions,
        recalls,
        f1_scores,
        macro_f1
    )

class_names = ["N", "SVEB", "VEB", "F"]

(
    train_accuracy,
    train_conf_matrix,
    train_precisions,
    train_recalls,
    train_f1_scores,
    train_macro_f1
) = evaluate_model(
    y_train_num,
    y_train_pred,
    class_names
)

print("\nTRAIN RESULTS")
print("------------------")

print(
    f"Accuracy: {train_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {train_macro_f1:.4f}"
)

print("\nConfusion Matrix:")
print(train_conf_matrix)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {train_precisions[i]:.4f} | "
        f"Recall: {train_recalls[i]:.4f} | "
        f"F1: {train_f1_scores[i]:.4f}"
    )

(
    val_accuracy,
    val_conf_matrix,
    val_precisions,
    val_recalls,
    val_f1_scores,
    val_macro_f1
) = evaluate_model(
    y_val_num,
    y_val_pred,
    class_names
)

print("\nVALIDATION RESULTS")
print("------------------")

print(
    f"Accuracy: {val_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {val_macro_f1:.4f}"
)

print("\nConfusion Matrix:")
print(val_conf_matrix)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {val_precisions[i]:.4f} | "
        f"Recall: {val_recalls[i]:.4f} | "
        f"F1: {val_f1_scores[i]:.4f}"
    )

W_2, b_2, loss_history_2 = train_logistic_regression(
    X_train_pca,
    y_train_num,
    n_classes=4,
    learning_rate=0.001,
    epochs=2000
)

y_train_pred_2 = predict(
    X_train_pca,
    W_2,
    b_2
)

y_val_pred_2 = predict(
    X_val_pca,
    W_2,
    b_2
)

train_accuracy_2 = np.mean(
    y_train_pred_2 == y_train_num
)

val_accuracy_2 = np.mean(
    y_val_pred_2 == y_val_num
)

(
    train_accuracy_2,
    train_conf_matrix_2,
    train_precisions_2,
    train_recalls_2,
    train_f1_scores_2,
    train_macro_f1_2
) = evaluate_model(
    y_train_num,
    y_train_pred_2,
    class_names
)

print("\nTRAIN RESULTS")
print("------------------")

print(
    f"Accuracy: {train_accuracy_2 * 100:.2f}%"
)

print(
    f"Macro F1: {train_macro_f1_2:.4f}"
)

print("\nConfusion Matrix:")
print(train_conf_matrix_2)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {train_precisions_2[i]:.4f} | "
        f"Recall: {train_recalls_2[i]:.4f} | "
        f"F1: {train_f1_scores_2[i]:.4f}"
    )

(
    val_accuracy_2,
    val_conf_matrix_2,
    val_precisions_2,
    val_recalls_2,
    val_f1_scores_2,
    val_macro_f1_2
) = evaluate_model(
    y_val_num,
    y_val_pred_2,
    class_names
)

print("\nVALIDATION RESULTS")
print("------------------")

print(
    f"Accuracy: {val_accuracy_2 * 100:.2f}%"
)

print(
    f"Macro F1: {val_macro_f1_2:.4f}"
)

print("\nConfusion Matrix:")
print(val_conf_matrix_2)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {val_precisions_2[i]:.4f} | "
        f"Recall: {val_recalls_2[i]:.4f} | "
        f"F1: {val_f1_scores_2[i]:.4f}"
    )

y_test_pred = predict(
    X_test_pca,
    W,
    b
)

test_accuracy = np.mean(
    y_test_pred == y_test_num
)

(
    test_accuracy,
    test_conf_matrix,
    test_precisions,
    test_recalls,
    test_f1_scores,
    test_macro_f1
) = evaluate_model(
    y_test_num,
    y_test_pred,
    class_names
)

print("\nTEST RESULTS - FIRST MODEL")
print("------------------")

print(
    f"Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {test_macro_f1:.4f}"
)

print("\nConfusion Matrix:")
print(test_conf_matrix)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {test_precisions[i]:.4f} | "
        f"Recall: {test_recalls[i]:.4f} | "
        f"F1: {test_f1_scores[i]:.4f}"
    )

def get_split_metrics(y_true, y_pred, class_names):

    (
        accuracy,
        conf_matrix,
        precisions,
        recalls,
        f1_scores,
        macro_f1
    ) = evaluate_model(
        y_true,
        y_pred,
        class_names
    )

    return {
        "Accuracy": accuracy,
        "Macro Precision": np.mean(precisions),
        "Macro Recall": np.mean(recalls),
        "Macro F1": macro_f1
    }

from imblearn.over_sampling import RandomOverSampler

oversampler = RandomOverSampler(
    sampling_strategy="auto",
    random_state=42
)

X_train_over, y_train_over = oversampler.fit_resample(
    X_train_scaled,
    y_train_num
)

print("Original training distribution:")
print(np.bincount(y_train_num))

print("\nOversampled training distribution:")
print(np.bincount(y_train_over))

# !pip install xgboost

from xgboost import XGBClassifier

xgb_over = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_over.fit(
    X_train_over,
    y_train_over
)

xgb_train_pred_over = xgb_over.predict(
    X_train_scaled
)

xgb_val_pred_over = xgb_over.predict(
    X_val_scaled
)

xgb_train_metrics = get_split_metrics(
    y_train_num,
    xgb_train_pred_over,
    class_names
)

xgb_val_metrics = get_split_metrics(
    y_val_num,
    xgb_val_pred_over,
    class_names
)

xgb_results_table = pd.DataFrame(
    [
        xgb_train_metrics,
        xgb_val_metrics,
    ],
    index=[
        "Training",
        "Validation",
    ]
)

print(
    "\nOVERSAMPLED XGBOOST"
)

print(
    xgb_results_table.round(4)
)

(
    train_accuracy_over,
    train_conf_matrix_over,
    train_precisions_over,
    train_recalls_over,
    train_f1_scores_over,
    train_macro_f1_over
) = evaluate_model(
    y_train_num,
    xgb_train_pred_over,
    class_names
)

print("\nTRAIN RESULTS - OVERSAMPLING")
print("------------------")

print(
    f"Accuracy: {train_accuracy_over * 100:.2f}%"
)

print(
    f"Macro F1: {train_macro_f1_over:.4f}"
)

print("\nConfusion Matrix:")
print(train_conf_matrix_over)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {train_precisions_over[i]:.4f} | "
        f"Recall: {train_recalls_over[i]:.4f} | "
        f"F1: {train_f1_scores_over[i]:.4f}"
    )

(
    val_accuracy_over,
    val_conf_matrix_over,
    val_precisions_over,
    val_recalls_over,
    val_f1_scores_over,
    val_macro_f1_over
) = evaluate_model(
    y_val_num,
    xgb_val_pred_over,
    class_names
)

print("\nVALIDATION RESULTS - OVERSAMPLING")
print("------------------")

print(
    f"Accuracy: {val_accuracy_over * 100:.2f}%"
)

print(
    f"Macro F1: {val_macro_f1_over:.4f}"
)

print("\nConfusion Matrix:")
print(val_conf_matrix_over)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {val_precisions_over[i]:.4f} | "
        f"Recall: {val_recalls_over[i]:.4f} | "
        f"F1: {val_f1_scores_over[i]:.4f}"
    )

from xgboost import XGBClassifier

regularized_configs = [
    # depth, lambda, alpha, gamma, min_child_weight
    (6, 1, 0,   0,   1),
    (5, 5, 0,   0,   2),
    (4, 5, 0.1, 0.1, 3),
    (4, 10, 0.5, 0.2, 5),
    (3, 10, 1,   0.5, 5)
]

over_regularized_results = []

for depth, reg_lambda, reg_alpha, gamma, min_child_weight in regularized_configs:

    model_over = XGBClassifier(
        n_estimators=300,
        max_depth=depth,
        learning_rate=0.1,

        # Sampling regularization
        subsample=0.8,
        colsample_bytree=0.8,

        # Explicit regularization
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        gamma=gamma,
        min_child_weight=min_child_weight,

        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model_over.fit(
        X_train_over,
        y_train_over
    )

    train_pred = model_over.predict(X_train_scaled)
    val_pred = model_over.predict(X_val_scaled)

    train_metrics = get_split_metrics(
        y_train_num,
        train_pred,
        class_names
    )

    val_metrics = get_split_metrics(
        y_val_num,
        val_pred,
        class_names
    )

    over_regularized_results.append({
        "depth": depth,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "gamma": gamma,
        "min_child_weight": min_child_weight,

        "train_accuracy": train_metrics["Accuracy"],
        "val_accuracy": val_metrics["Accuracy"],

        "train_macro_f1": train_metrics["Macro F1"],
        "val_macro_f1": val_metrics["Macro F1"],

        "f1_gap":
            train_metrics["Macro F1"]
            - val_metrics["Macro F1"]
    })

regularized_table = pd.DataFrame(
    over_regularized_results
)

print(
    regularized_table.round(4).to_string(
        index=False
    )
)

from imblearn.over_sampling import SMOTE
from collections import Counter

sm = SMOTE(random_state=42, k_neighbors=1)
print('Before SMOTE resampled dataset shape %s' % Counter(y_train_num))
X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train_num)
print('SMOTE resampled dataset shape %s' % Counter(y_train_smote))

from xgboost import XGBClassifier

xgb_smote = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_smote.fit(
    X_train_smote,
    y_train_smote
)

xgb_train_pred_smote = xgb_smote.predict(
    X_train_scaled
)

xgb_val_pred_smote = xgb_smote.predict(
    X_val_scaled
)

xgb_train_metrics_smote = get_split_metrics(
    y_train_num,
    xgb_train_pred_smote,
    class_names
)

xgb_val_metrics_smote = get_split_metrics(
    y_val_num,
    xgb_val_pred_smote,
    class_names
)

xgb_results_table = pd.DataFrame(
    [
        xgb_train_metrics_smote,
        xgb_val_metrics_smote,
    ],
    index=[
        "Training",
        "Validation",
    ]
)

print(
    "\nSMOTE XGBOOST"
)

print(
    xgb_results_table.round(4)
)

(
    train_accuracy_smote,
    train_conf_matrix_smote,
    train_precisions_smote,
    train_recalls_smote,
    train_f1_scores_smote,
    train_macro_f1_smote
) = evaluate_model(
    y_train_num,
    xgb_train_pred_smote,
    class_names
)

print("\nTRAIN RESULTS - SMOTE")
print("------------------")

print(
    f"Accuracy: {train_accuracy_smote * 100:.2f}%"
)

print(
    f"Macro F1: {train_macro_f1_smote:.4f}"
)

print("\nConfusion Matrix:")
print(train_conf_matrix_smote)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {train_precisions_smote[i]:.4f} | "
        f"Recall: {train_recalls_smote[i]:.4f} | "
        f"F1: {train_f1_scores_smote[i]:.4f}"
    )

(
    val_accuracy_smote,
    val_conf_matrix_smote,
    val_precisions_smote,
    val_recalls_smote,
    val_f1_scores_smote,
    val_macro_f1_smote
) = evaluate_model(
    y_val_num,
    xgb_val_pred_smote,
    class_names
)

print("\n VALIDATION RESULTS - SMOTE")
print("------------------")

print(
    f"Accuracy: {val_accuracy_smote * 100:.2f}%"
)

print(
    f"Macro F1: {val_macro_f1_smote:.4f}"
)

print("\nConfusion Matrix:")
print(val_conf_matrix_smote)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {val_precisions_smote[i]:.4f} | "
        f"Recall: {val_recalls_smote[i]:.4f} | "
        f"F1: {val_f1_scores_smote[i]:.4f}"
    )

from xgboost import XGBClassifier

regularized_configs = [
    # depth, lambda, alpha, gamma, min_child_weight
    (6, 1, 0,   0,   1),
    (5, 5, 0,   0,   2),
    (4, 5, 0.1, 0.1, 3),
    (4, 10, 0.5, 0.2, 5),
    (3, 10, 1,   0.5, 5)
]

smote_regularized_results = []

for depth, reg_lambda, reg_alpha, gamma, min_child_weight in regularized_configs:

    model_smote = XGBClassifier(
        n_estimators=300,
        max_depth=depth,
        learning_rate=0.1,

        # Sampling regularization
        subsample=0.8,
        colsample_bytree=0.8,

        # Explicit regularization
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        gamma=gamma,
        min_child_weight=min_child_weight,

        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model_smote.fit(
        X_train_smote,
        y_train_smote
    )

    train_pred_smote = model_smote.predict(X_train_scaled)
    val_pred_smote = model_smote.predict(X_val_scaled)

    train_metrics_smote = get_split_metrics(
        y_train_num,
        train_pred_smote,
        class_names
    )

    val_metrics_smote = get_split_metrics(
        y_val_num,
        val_pred_smote,
        class_names
    )

    smote_regularized_results.append({
        "depth": depth,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "gamma": gamma,
        "min_child_weight": min_child_weight,

        "train_accuracy": train_metrics_smote["Accuracy"],
        "val_accuracy": val_metrics_smote["Accuracy"],

        "train_macro_f1": train_metrics_smote["Macro F1"],
        "val_macro_f1": val_metrics_smote["Macro F1"],

        "f1_gap":
            train_metrics_smote["Macro F1"]
            - val_metrics_smote["Macro F1"]
    })

smote_regularized_table = pd.DataFrame(
    smote_regularized_results
)

print(
    smote_regularized_table.round(4).to_string(
        index=False
    )
)

xgb_test_pred_smote = xgb_smote.predict(
    X_test_scaled
)

xgb_test_metrics_smote = get_split_metrics(
    y_test_num,
    xgb_test_pred_smote,
    class_names
)

xgb_test_table_smote = pd.DataFrame(
    [xgb_test_metrics_smote],
    index=["Test"]
)

print("\nSMOTE XGBOOST - FINAL TEST")
print(xgb_test_table_smote.round(4))

(
    test_accuracy_smote,
    test_conf_matrix_smote,
    test_precisions_smote,
    test_recalls_smote,
    test_f1_scores_smote,
    test_macro_f1_smote
) = evaluate_model(
    y_test_num,
    xgb_test_pred_smote,
    class_names
)

print("\nFINAL TEST RESULTS - SMOTE XGBOOST")
print("----------------------------------")
print(f"Accuracy: {test_accuracy_smote * 100:.2f}%")
print(f"Macro F1: {test_macro_f1_smote:.4f}")

print("\nConfusion Matrix:")
print(test_conf_matrix_smote)

for i in range(len(class_names)):
    print(
        f"{class_names[i]} | "
        f"Precision: {test_precisions_smote[i]:.4f} | "
        f"Recall: {test_recalls_smote[i]:.4f} | "
        f"F1: {test_f1_scores_smote[i]:.4f}"
    )