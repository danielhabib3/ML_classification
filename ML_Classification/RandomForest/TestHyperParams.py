from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
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

# Manual test of hyperparameters
for n_estimators in [50, 100, 200]:
    for max_depth in [None, 5, 10]:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=31
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(
            f"n_estimators={n_estimators}, max_depth={max_depth} => Accuracy: {acc}")
        print("Classification Report:")
        print(classification_report(
            y_val, y_pred, target_names=["DOWN", "UP"]))
