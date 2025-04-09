# Réseaux Neurones

Ce projet explore l'utilisation des réseaux de neurones pour résoudre des problèmes de classification en apprentissage automatique. Ce document explique comment utiliser les fichiers présents dans ce répertoire.

## Contenu des Fichiers

- **`data_processing.py`** : Script pour le prétraitement des données avant l'entraînement.
- **`generateSubmission.py`** : Script pour générer des fichiers de soumission à partir des prédictions du modèle.
- **`modelstestHyperParams.py`** : Script pour tester différents hyperparamètres des modèles.
- **`train_model.py`** : Script principal pour entraîner les modèles de réseaux de neurones.

## Instructions d'Utilisation

1. **Préparation des Données** :
    - Assurez-vous que vos données sont prêtes et correctement formatées pour le prétraitement.

2. **Prétraitement des Données** :
    - Exécutez le script `data_processing.py` pour préparer les données :
      ```bash
      python data_processing.py
      ```

3. **Entraînement du Modèle** :
    - Lancez le script `train_model.py` pour entraîner un modèle :
      ```bash
      python train_model.py
      ```

4. **Test des Hyperparamètres** :
    - Utilisez le script `modelstestHyperParams.py` pour expérimenter avec différents hyperparamètres :
      ```bash
      python modelstestHyperParams.py
      ```

5. **Génération de Soumissions** :
    - Exécutez le script `generateSubmission.py` pour créer des fichiers de soumission :
      ```bash
      python generateSubmission.py
      ```

## Structure du Projet

- `data_processing.py` : Prétraitement des données.
- `train_model.py` : Entraînement des modèles.
- `modelstestHyperParams.py` : Tests des hyperparamètres.
- `generateSubmission.py` : Génération des fichiers de soumission.

## Remarques

- Assurez-vous que les données sont correctement formatées avant de lancer les scripts.
- Vous pouvez ajuster les paramètres dans les scripts pour expérimenter avec différents réglages.

## Contributeurs

- **Daniel Habib** - Développeur principal.

## Licence

Ce projet est sous licence MIT. Consultez le fichier `LICENSE` pour plus d'informations.