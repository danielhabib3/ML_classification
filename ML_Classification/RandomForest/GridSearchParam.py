import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the dataset
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data",
                          "classification", "train.csv")

# Load dataset
df = pd.read_csv(TRAIN_PATH)
df = df.sort_values(by=["date", "hour"]).reset_index(drop=True)

# Encode target
df["bc_price_evo"] = LabelEncoder().fit_transform(df["bc_price_evo"])

# Split features/target
X = df.drop(columns=["id", "bc_price_evo"])
y = df["bc_price_evo"]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Grid search
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=31),
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1
)

grid.fit(X_train, y_train)

# Results
print("Best parameters found:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)
print("Test set accuracy:", grid.score(X_val, y_val))
# y_pred = grid.predict(X_val)  # Uncomment to get predictions on validation set
# accuracy = accuracy_score(y_val, y_pred)  # Uncomment to calculate accuracy
# print("Validation set accuracy:", accuracy)  # Uncomment to print accuracy
