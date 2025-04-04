import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv('train.csv')

for col in ['urb_cons', 'exturb_cons', 'overall_cons', 'nox']:
    mean = data[col].mean()
    std = data[col].std()
    data = data[(data[col] >= mean - 2*std) & (data[col] <= mean + 2*std)]

data['hc'] = data['hc'].fillna(data['hcnox'] - data['nox'])
data['hcnox'] = data['hcnox'].fillna(data['nox'] + data['hc'])

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

features = ['max_power', 'weight_min', 'weight_max', 'urb_cons', 'exturb_cons',
            'overall_cons', 'co', 'hc', 'nox', 'ptcl']
target = 'co2'

X = data[features]
y = data[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train_imputed, y_train)

y_pred = model.predict(X_val_imputed)
print("Mean Squared Error:", mean_squared_error(y_val, y_pred))
print("R² Score:", r2_score(y_val, y_pred))

test = pd.read_csv('test.csv')

test['hc'] = test['hc'].fillna(test['hcnox'] - test['nox'])
test['hcnox'] = test['hcnox'].fillna(test['nox'] + test['hc'])


test_features = test[features].copy()
test_features_imputed = imputer.transform(test_features)

co2_pred = model.predict(test_features_imputed)
test['co2'] = co2_pred.round().astype(int)

test[['id', 'co2']].to_csv('submission.csv', index=False)
