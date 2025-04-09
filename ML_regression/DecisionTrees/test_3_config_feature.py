import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "regression", "train.csv")
df = pd.read_csv(TRAIN_PATH)

configs = {
    "all": [],
    "drop_consumptions": ['urb_cons', 'exturb_cons', 'overall_cons'],
    "drop_pollutants": ['co', 'nox', 'ptcl', 'hcnox']
}

for name, drops in configs.items():
    df_config = df.drop(['id', 'brand', 'model', 'car_class',
                        'grbx_type_ratios'] + drops, axis=1, errors="ignore")
    df_config = pd.get_dummies(
        df_config, columns=['fuel_type', 'range', 'hybrid'], drop_first=True)
    df_config = df_config.fillna(df_config.median())

    X = df_config.drop("co2", axis=1)
    y = df_config["co2"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print(f"Config: {name} → MAE: {mean_absolute_error(y_val, y_pred)}")
