# RandomForest Readme

Ce répertoire contient deux fichiers principaux liés à l'utilisation de l'algorithme Random Forest pour la classification :

## 1. `randomForestTrain.py`
- **Description** : Ce script implémente l'entraînement d'un modèle Random Forest. Il charge les données, effectue un prétraitement, ajuste les hyperparamètres et entraîne le modèle.
- **Utilisation** : Exécutez ce fichier pour entraîner un modèle sur vos données. Assurez-vous que les données d'entraînement sont placées dans le répertoire `data` et correctement formatées.

## 2. `randomForestTest.py`
- **Description** : Ce script utilise un modèle Random Forest préalablement entraîné pour effectuer des prédictions sur de nouvelles données.
- **Utilisation** : Placez les données à prédire dans le répertoire `data` et exécutez ce script pour obtenir les résultats.

## Prérequis
- Python 3.x
- Bibliothèques nécessaires : `scikit-learn`, `pandas`, `numpy`

## Instructions
1. Placez vos fichiers de données dans le répertoire `data`.
2. Exécutez `randomForestTrain.py` pour entraîner le modèle.
3. Utilisez `randomForestTest.py` pour effectuer des prédictions.

