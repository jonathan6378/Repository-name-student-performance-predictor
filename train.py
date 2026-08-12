import os
import pickle
import numpy as np
import pandas as pd

from random_forest import RandomForestRegressorScratch

DATA_PATH = "data/student_data.csv"
MODEL_PATH = "model/random_forest.pkl"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
    "extracurricular"
]

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].values.astype(float)
y = df["final_score"].values.astype(float)

# Manual train/test split
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
split = int(0.80 * len(X))

train_idx = indices[:split]
test_idx = indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = RandomForestRegressorScratch(
    n_estimators=30,
    max_depth=8,
    min_samples_split=4,
    max_features="sqrt",
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
ss_res = np.sum((y_test - predictions) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - ss_res / ss_tot

print("Random Forest Regressor — From Scratch")
print("---------------------------------------")
print(f"Trees : {model.n_estimators}")
print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.3f}")

print("\nFeature Importance:")
for feature, importance in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature:25s} {importance:.3f}")

os.makedirs("model", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "features": FEATURES}, f)

print(f"\nSaved model to {MODEL_PATH}")
