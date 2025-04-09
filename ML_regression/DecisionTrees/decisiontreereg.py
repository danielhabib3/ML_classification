import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "regression", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "regression", "test.csv")


def preprocess_data(train_df, test_df):
    combined = pd.concat([train_df, test_df])
    combined = combined.drop(
        ['id', 'brand', 'model', 'car_class', 'grbx_type_ratios'], axis=1)
    combined = pd.get_dummies(
        combined, columns=['fuel_type', 'range', 'hybrid'], drop_first=True)
    combined = combined.fillna(combined.median())

    train_processed = combined.iloc[:len(train_df), :]
    test_processed = combined.iloc[len(train_df):, :]

    X_train = train_processed.drop("co2", axis=1)
    y_train = train_processed["co2"]
    X_test = test_processed.drop("co2", axis=1, errors="ignore")
    return X_train, y_train, X_test


# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
X_train, y_train, X_test = preprocess_data(train_df, test_df)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Eval
val_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, val_pred)
print("Train MAE:", mae)

# Submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "co2": model.predict(X_test)
})
submission.to_csv(os.path.join(BASE_DIR, "submission.csv"), index=False)
print("Submission saved.")
