import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "regression", "train.csv")
df = pd.read_csv(TRAIN_PATH)

df = df.drop(['id', 'brand', 'model', 'car_class', 'grbx_type_ratios'], axis=1)
df = pd.get_dummies(
    df, columns=['fuel_type', 'range', 'hybrid'], drop_first=True)
df = df.fillna(df.median())

X = df.drop("co2", axis=1)
y = df["co2"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

for depth in [3, 5, 10, 20, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"max_depth = {depth} → MAE = {mae}")
