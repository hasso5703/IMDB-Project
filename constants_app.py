import pandas as pd
# import pickle
from typing import Final

from constants import FICHIER_COMMENTAIRE_PLONGEMENT, FICHIER_RESEAU_NEURONE
from language_model import ModeleLangage
from tensorflow.keras.models import load_model

"""
===========================================================================
constantes globales spécifiques à l'application principale
"""

# dataframe commentaire et plongement
DF_COMMENTAIRE_PLONGEMENT: Final[pd.DataFrame] = pd.read_csv(FICHIER_COMMENTAIRE_PLONGEMENT)
# NB_MODALITES = len(set(DF_COMMENTAIRE_PLONGEMENT['titre'].values))

# BERT
MODELE_LANGAGE: Final[ModeleLangage] = ModeleLangage()

# Chargez la liste depuis le fichier binaire
# with open(str(FICHIER_FILMS_LIST), 'rb') as fichier:
    # FILMS_LIST = pickle.load(fichier)

# Charger le modèle Keras
RESEAU_NEURONE = load_model(FICHIER_RESEAU_NEURONE)

# label pour le commentaire de l'utilisateur
LABEL_COMMENTAIRE_UTILISATEUR: Final[str] = '|---USER---|'

LISTE_FILMS = sorted(list(set(DF_COMMENTAIRE_PLONGEMENT["titre"])))
LISTE_FILMS_INITIAL = LISTE_FILMS[:5]
# print("len liste films ", len(LISTE_FILMS))
NB_MODALITES = len(LISTE_FILMS)
