# Random Forest Classification

This folder contains all experiments related to the Random Forest model for binary classification (electricity price UP or DOWN).

## Files
- `RandomForestBasic.py`: Basic implementation with default parameters
- `RandomForestWithParams.py`: Model using tuned parameters
- `GridSearchParam.py`: Hyperparameter search using GridSearchCV
- `TestHyperParams.py`: Manual hyperparameter testing

## Dataset
All scripts assume the data is located in `../data/train.csv`. The target variable is `bc_price_evo`.

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
