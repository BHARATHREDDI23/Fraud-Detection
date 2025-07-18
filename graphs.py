import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")
print("\n✅ Loaded Model: fraud_detection_model.pkl")

# Load test dataset
df_test = pd.read_csv(r"C:\Users\kiran\OneDrive\Desktop\onlinefraud.csv")  # Update the path if needed

# Take only the first 10,000 rows
df_test = df_test.head(10000)

# Encode 'type' column using one-hot encoding
df_test = pd.get_dummies(df_test, columns=["type"], drop_first=True)

# Separate features and target
X_test = df_test.drop(columns=["isFraud"])
y_test = df_test["isFraud"]

# Ensure test data has same features as the model
model_features = model.get_booster().feature_names
missing_cols = set(model_features) - set(X_test.columns)
extra_cols = set(X_test.columns) - set(model_features)

# Add missing columns with zeros
for col in missing_cols:
    X_test[col] = 0

# Drop extra columns
X_test = X_test[model_features]

# Get fraud probabilities
fraud_probs = model.predict_proba(X_test)[:, 1]

# Apply the new fraud threshold of 0.61
y_pred = (fraud_probs >= 0.61).astype(int)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, fraud_probs)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n📊 Model Evaluation:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")
print(f"\n🔹 Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, fraud_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Plot Accuracy as a Bar Chart
plt.figure(figsize=(4, 3))
plt.bar(["Accuracy"], [accuracy], color="green")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Accuracy")
plt.show()
