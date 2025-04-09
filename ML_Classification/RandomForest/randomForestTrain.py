import sys
sys.path.append("..")
import load_data as l
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_train, df_test = l.load_data_train("../data/regression/")

y_test = df_test['co2']

# Separate features and target variable in df_train
X_train = df_train.drop(columns=['co2'])
y_train = df_train['co2']

# Prepare features in df_test
X_test = df_test.drop(columns=['co2'])

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

# Predict co2 in df_test
y_pred = rf.predict(X_test)

# Calculate the squared distance (mean squared error)
squared_distance = mean_squared_error(y_test, y_pred)
print("Squared Distance (Train):", squared_distance)

