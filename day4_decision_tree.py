import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

# Step 1 — Data load
iris = load_iris()
X = iris.data
y = iris.target
custom_names = ['Rose', 'Sunflower', 'Lotus']

# Step 2 — Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3 — Model
model = DecisionTreeClassifier(
    criterion='gini', max_depth=4,
    min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# Step 4 — Predict
y_pred = model.predict(X_test)

# Step 5 — Evaluate
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=custom_names))

# Step 6 — Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=custom_names).plot(ax=axes[0])

# Step 7 — Tree visualize
plot_tree(model, feature_names=iris.feature_names,
          class_names=custom_names,
          filled=True, rounded=True, ax=axes[1])
plt.tight_layout()          # graph ko fit karta hai
plt.savefig("day4_results.png", dpi=150)
plt.show()                  # graph screen par dikhata hai


# Step 8 — Overfitting experiment
for depth in [1, 2, 3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    print(depth, dt.score(X_train, y_train), dt.score(X_test, y_test))

# Step 9 — Feature importance
for feat, imp in zip(iris.feature_names, model.feature_importances_):
    print(f"{feat}: {imp:.4f}")

# Step 10 — New sample predict
sample = np.array([[5.1, 3.5, 1.4, 0.2]])


print(custom_names[model.predict(sample)[0]])  # Rose/Sunflower/Lotus dikhega

print(model.predict_proba(sample))