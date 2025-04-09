import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
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

# Tuned Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=31
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

# Evaluation
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["DOWN", "UP"]))
