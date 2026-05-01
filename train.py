# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =========================
# CREATE FOLDERS (IMPORTANT)
# =========================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =========================
# LOAD DATA
# =========================
file_path = "mitbih_dataset.csv"
df = pd.read_csv(file_path)

print("Data shape:", df.shape)
print(df.head())

# =========================
# LABEL CREATION
# =========================
df['label'] = df['type'].apply(lambda x: 0 if x == 'N' else 1)

print("\nBinary Class distribution:")
print(df['label'].value_counts())

# =========================
# DROP UNUSED COLUMNS
# =========================
if 'record' in df.columns:
    df = df.drop(columns=['record'])

df = df.drop(columns=['type'])

# =========================
# HANDLE MISSING VALUES
# =========================
df = df.fillna(df.mean())

# =========================
# FEATURES & TARGET
# =========================
X = df.drop(columns=['label'])
y = df['label']

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced')
}

results = {}

# =========================
# TRAIN & EVALUATE
# =========================
for name, model in models.items():
    print(f"\n========== {name} ==========")

    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results[name] = f1

    print("F1 Score:", f1)
    print("ROC-AUC:", roc)
    print(classification_report(y_test, y_pred))

# =========================
# MODEL COMPARISON
# =========================
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values())
plt.title("Model Comparison (F1-score)")
plt.xticks(rotation=30)
plt.savefig("results/model_comparison.png")
plt.show()

# =========================
# BEST MODEL SELECTION
# =========================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# Predict best model
if best_model_name in ["Logistic Regression", "KNN", "SVM"]:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig("results/confusion_matrix.png")
plt.show()

# =========================
# FEATURE IMPORTANCE (RF)
# =========================
rf = models["Random Forest"]

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:")
print(feat_imp.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp.head(10))
plt.title("Top ECG Features")
plt.savefig("results/feature_importance.png")
plt.show()

# =========================
# SAVE MODELS
# =========================
for name, model in models.items():
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{filename}")

# save scaler
joblib.dump(scaler, "models/scaler.pkl")

# save best model separately (IMPORTANT FOR DEPLOYMENT)
best_filename = f"models/best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
joblib.dump(best_model, best_filename)

print("\nAll models + scaler + best model saved successfully!")