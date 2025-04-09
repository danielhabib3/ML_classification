# RandomForest Readme

Ce répertoire contient deux fichiers principaux liés à l'utilisation de l'algorithme Random Forest pour la classification :

## 1. `randomForestTrain.py`
- **Description** : Ce script implémente l'entraînement d'un modèle Random Forest. Il charge les données, effectue un prétraitement, ajuste les hyperparamètres et entraîne le modèle.
- **Utilisation** : Exécutez ce fichier pour entraîner un modèle sur vos données. Assurez-vous que les données d'entraînement sont placées dans le répertoire `data` et correctement formatées.

## 2. `randomForestTest.py`
- **Description** : Ce script utilise un modèle Random Forest préalablement entraîné pour effectuer des prédictions sur de nouvelles données.
- **Utilisation** : Placez les données à prédire dans le répertoire `data` et exécutez ce script pour obtenir les résultats.

## 3. `TestHyperParams.py`
- **Description** : Ce script teste manuellement différentes combinaisons de paramètres (`n_estimators` et `max_depth`) pour le modèle Random Forest. Il affiche pour chaque combinaison l’accuracy sur les données de validation ainsi qu’un rapport de classification.
- **Utilisation** : Exécutez ce script pour explorer rapidement les performances du modèle selon plusieurs hyperparamètres standards.

## 4. `GridSearchParam.py`
- **Description** : Ce script effectue une recherche systématique d'hyperparamètres via `GridSearchCV` de `scikit-learn`. Il explore un espace de paramètres plus large et utilise une validation croisée pour identifier la meilleure configuration.
- **Utilisation** : Lancez ce script pour obtenir automatiquement les meilleurs hyperparamètres et afficher leur score de validation croisée ainsi que l’accuracy sur le jeu de test.

## Prérequis
- Python 3.x
- Bibliothèques nécessaires : `scikit-learn`, `pandas`, `numpy`

## Instructions
1. Placez vos fichiers de données dans le répertoire `data`.
2. Exécutez `randomForestTrain.py` pour entraîner le modèle.
3. Utilisez `randomForestTest.py` pour effectuer des prédictions.

