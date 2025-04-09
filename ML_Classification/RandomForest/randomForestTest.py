import sys
sys.path.append("..")
import load_data as l
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd



df_train, df_test = l.load_data_train("../data/regression/", 1)

# Separate features and target variable in df_train
X_train = df_train.drop(columns=['co2'])
y_train = df_train['co2']


# Train a Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,  # Number of trees in the forest
    criterion='squared_error',  # default for regression tasks
    max_depth=None,    # maximum depth of the tree
    min_samples_split=2,  # minimum samples required to split an internal node
    min_samples_leaf=1,   # minimum samples required to be at a leaf node
    min_weight_fraction_leaf=0.0,  # minimum weighted fraction of the sum total of weights at a leaf node
    max_features='sqrt',    # number of features to consider when looking for the best split
    random_state=0,       # seed for reproducibility
    max_leaf_nodes=None,  # maximum number of leaf nodes
    min_impurity_decrease=0.0,  # minimum impurity decrease required for a split
)
rf.fit(X_train, y_train)





df = l.load_data_test("../data/regression/")


df_id = df['id']

df = df.drop(columns=['id'])

# Predict co2 in df_test
y_pred = rf.predict(df)

# Round y_pred to the nearest integer
y_pred = y_pred.round().astype(int)

# Create a DataFrame with df_id and y_pred
result_df = pd.DataFrame({
    'id': df_id,
    'co2': y_pred
})

# Save the result to a CSV file
result_df.to_csv("predictions.csv", index=False)