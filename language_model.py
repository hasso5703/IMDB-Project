import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from constants import NOM_MODELE


class ModeleLangage:
    nom: str
    modele: SentenceTransformer

    def __init__(self, nom: str = NOM_MODELE) -> None:
        self.nom = nom
        self.modele = SentenceTransformer(nom)

    @property
    def longueur_sequence(self) -> int:
        return self.modele.max_seq_length

    @property
    def dimension_plongement(self) -> int:
        return self.modele.get_sentence_embedding_dimension()

    def __call__(self, liste_chaine: str | list[str] | pd.Series, barre_progression: bool = False) -> pd.DataFrame:

        if isinstance(liste_chaine, pd.Series):
            x = liste_chaine.values
        else:
            x = liste_chaine

        with torch.no_grad():  # optimisation: les gradients ne sont pas calculés
            plongement = self.modele.encode(x, convert_to_numpy=True, show_progress_bar=barre_progression)

        if isinstance(liste_chaine, str):
            plongement = np.reshape(plongement, (1, -1))

        df = pd.DataFrame(plongement, columns=[f'V_{i:03d}' for i in range(plongement.shape[1])])

        if isinstance(liste_chaine, pd.Series):
            df.index = liste_chaine.index

        return df
