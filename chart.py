import textwrap
from typing import Dict, Any, Tuple

import numpy as np
# import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from sklearn.manifold import TSNE
# from sklearn.neighbors import KNeighborsClassifier
# from constants_app import LABEL_COMMENTAIRE_UTILISATEUR
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def generer_graphique_projection(commentaire_plongement: pd.DataFrame,
                                 perplexite: float,
                                 distance: str,
                                 selected_method: str) -> Figure:
    if commentaire_plongement.shape[0] < 10:
        return {}
    perplexite = min(perplexite, commentaire_plongement.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexite, metric=distance, learning_rate='auto', init='random')
    # pca = PCA(n_components=2)
    coord = tsne.fit_transform(commentaire_plongement.loc[:, 'V_000':])
    # coord_pca = pca.fit(commentaire_plongement.loc[:, 'V_000':])
    # coord_titre_pca = pd.DataFrame(coord_pca, columns=['x', 'y'], index=commentaire_plongement.index)
    coord_titre = pd.DataFrame(coord, columns=['x', 'y'], index=commentaire_plongement.index)
    coord_titre['titre'] = commentaire_plongement.titre.values
    # coord_titre_pca["titre"] = commentaire_plongement.titre.values
    # coord_titre_pca['review'] = commentaire_plongement.review.apply(
    #    lambda chaine: '<br>'.join(textwrap.wrap(textwrap.shorten(chaine, width=300), width=50)))
    coord_titre['review'] = commentaire_plongement.review.apply(
        lambda chaine: '<br>'.join(textwrap.wrap(textwrap.shorten(chaine, width=300), width=50)))
    # coord_titre['utilisateur'] = coord_titre.titre.apply(lambda chaine: 'u' if chaine == LABEL_ENTREE_UTILISATEUR
    # else 'c')
    figure = px.scatter(data_frame=coord_titre, x='x', y='y', color='titre', labels={'x': '', 'y': ''},
                        hover_name='titre', hover_data={'titre': False, 'x': False, 'y': False, 'review': True})
    # figure_pca = px.scatter(data_frame=coord_titre_pca, x='x', y='y', color='titre', labels={'x': '', 'y': ''},
    #                        hover_name='titre', hover_data={'titre': False, 'x': False, 'y': False, 'review': True})

    return figure


def generer_graphique_prediction_film(probabilite: pd.Series) -> go.Figure:
    return go.Figure(data=[go.Bar(x=probabilite.index, y=probabilite)])
