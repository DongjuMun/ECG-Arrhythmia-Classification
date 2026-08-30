import numpy as np
import pandas as pd

df = pd.read_csv("MIT-BIH Arrhythmia Database.csv")
print(df.head())

print(df.info())

df.describe()

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
df.head()

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

df.drop(ccolumns=["type", "record"]).hist(
    figsize=(15, 15),
    bins=30
)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

numeric_df = df.drop(columns=["type"])

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

"""Since I am going to use logistic regression, I will remove 1_qrs_morph0 and 0_qrs_morph0 and leave 1_qPeak and 0_qPeak   """

df = df.drop(columns=['0_qrs_morph0', '1_qrs_morph0'])
df.head()

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

# 80% of records for training
split_index = int(len(records) * 0.8)

train_records = records[:split_index]
test_records = records[split_index:]

# Create masks
train_mask = df["record"].isin(train_records)
test_mask = df["record"].isin(test_records)

# Split using the features DataFrame to ensure only numeric data
X_train = features[train_mask].to_numpy()
X_test = features[test_mask].to_numpy()

y_train = target[train_mask].to_numpy()
y_test = target[test_mask].to_numpy()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Train %:", len(X_train) / len(features) * 100)
print("Test %:", len(X_test) / len(features) * 100)

mean_train = np.mean(X_train, axis=0)
std_train = np.std(X_train, axis=0)
std_train[std_train == 0] = 1
X_train_scaled = (X_train - mean_train) / std_train
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
X_test_pca = X_test_scaled @ selected_eigenvectors

print(X_train_pca.shape)
print(X_test_pca.shape)

class_to_idx = {
    "N": 0,
    "SVEB": 1,
    "VEB": 2,
    "F": 3,
    "Q": 4
}

idx_to_class = {
    0: "N",
    1: "SVEB",
    2: "VEB",
    3: "F",
    4: "Q"
}

y_train_num = np.array([class_to_idx[label] for label in y_train])
y_test_num = np.array([class_to_idx[label] for label in y_test])

Y_train = np.eye(5)[y_train_num]
Y_test = np.eye(5)[y_test_num]
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
    n_classes=5,
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

y_test_pred = predict(
    X_test_pca,
    W,
    b
)

train_accuracy = np.mean(
    y_train_pred == y_train_num
)

test_accuracy = np.mean(
    y_test_pred == y_test_num
)

print(
    f"Train accuracy: {train_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy: {test_accuracy * 100:.2f}%"
)

conf_matrix = np.zeros((5, 5), dtype=int)

for true, pred in zip(y_test_num, y_test_pred):
    conf_matrix[true, pred] += 1

print(conf_matrix)

class_names = ["N", "SVEB", "VEB", "F", "Q"]
num_classes = len(class_names)

precisions = []
recalls = []
f1_scores = []

for i in range(num_classes):

    TP = conf_matrix[i, i]

    FP = np.sum(conf_matrix[:, i]) - TP
    FN = np.sum(conf_matrix[i, :]) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(
        f"{class_names[i]} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

macro_f1 = np.mean(f1_scores)

print("Macro F1:", macro_f1)