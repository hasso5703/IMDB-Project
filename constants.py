from typing import Final
from pathlib import Path

"""
===========================================================================
constantes globales à l'application principale et aux scripts secondaires
"""
DOSSIER_DATA: Final[str] = r'/Users/hasan/PycharmProjects/IMDB-Project/data'
DOSSIER_COMMENTAIRE: Final[str] = Path(DOSSIER_DATA) / 'reviews'
DOSSIER_PLONGEMENTS: Final[str] = Path(DOSSIER_DATA) / 'embeddings'
DOSSIER_MODELS: Final[str] = Path(DOSSIER_DATA) / 'models'

FICHIER_COMMENTAIRE_PLONGEMENT: Final[str] = Path(DOSSIER_PLONGEMENTS) / 'data_384.csv' # data_384.csv
FICHIER_RESEAU_NEURONE: Final[str] = Path(DOSSIER_MODELS) / 'best_model_384_14-02-2024_17-39-53.keras' # best_model_384_14-02-2024_17-39-53.keras
FICHIER_FILMS_LIST: Final[str] = Path(DOSSIER_DATA) / 'films_list.pkl'
FICHIER_RESEAU_NEURONE_RATING = Path(DOSSIER_MODELS) / 'rating_best_model.h5'
FICHIER_MATRICE_CONFUSION = Path(DOSSIER_MODELS) / 'conf_matrix.pkl'
FICHIER_RESEAU_NEURONE_MLP = Path(DOSSIER_MODELS) / 'commentaire_pmc.pickle'
MIN_MAX_MOT_INITIAL: Final[tuple[int, int]] = (100, 300)
NB_MAX_COMMENTAIRE_PAR_FILM_INITIAL: Final[int] = 55
PERPLEXITE_TSNE_INITIAL: Final[float] = 5
DISTANCE_TSNE_INITIAL: Final[str] = 'cosine'

NOM_MODELE: Final[str] = "all-MiniLM-L6-v2"  # all-mpnet-base-v2
