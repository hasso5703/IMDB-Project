import pickle
from typing import Final

import pandas as pd
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import load_model

from constants import FICHIER_COMMENTAIRE_PLONGEMENT, FICHIER_RESEAU_NEURONE, FICHIER_RESEAU_NEURONE_RATING, FICHIER_MATRICE_CONFUSION, FICHIER_RESEAU_NEURONE_MLP
from language_model import ModeleLangage

"""
===========================================================================
constantes globales spécifiques à l'application principale
"""

# dataframe commentaire et plongement
DF_COMMENTAIRE_PLONGEMENT: Final[pd.DataFrame] = pd.read_csv(FICHIER_COMMENTAIRE_PLONGEMENT)

# BERT
MODELE_LANGAGE: Final[ModeleLangage] = ModeleLangage()

K = 3

# Charger le modèle Keras
RESEAU_NEURONE = load_model(FICHIER_RESEAU_NEURONE)
RESEAU_NEURONE_RATING = load_model(FICHIER_RESEAU_NEURONE_RATING)

# label pour le commentaire de l'utilisateur
LABEL_COMMENTAIRE_UTILISATEUR: Final[str] = '|---USER---|'

LISTE_FILMS = sorted(list(set(DF_COMMENTAIRE_PLONGEMENT["titre"])))
LISTE_FILMS_INITIAL = LISTE_FILMS[:5]
NB_MODALITES = len(LISTE_FILMS)

# Charger la matrice de confusion ultérieurement
with open(FICHIER_MATRICE_CONFUSION, 'rb') as f:
    conf_matrix = pickle.load(f)

TP = conf_matrix[1][1]
FP = conf_matrix[0][1]
TN = conf_matrix[0][0]
FN = conf_matrix[1][0]

# Calculer TPR et FPR
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)

conf_matrix = [[TN, FP], [FN, TP]]

# chargement du MLP pour la classification
with open(FICHIER_RESEAU_NEURONE_MLP, 'rb') as fichier:
    dico = pickle.load(fichier)

RESEAU_NEURONE_MLP: Final[MLPClassifier] = dico['reseau_neurone']