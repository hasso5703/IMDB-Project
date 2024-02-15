import textwrap
# import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

# from constants_app import LABEL_COMMENTAIRE_UTILISATEUR


def generer_graphique_projection(commentaire_plongement: pd.DataFrame, perplexite: float, distance: str) -> go.Figure:
    if commentaire_plongement.shape[0] < 10:
        return {}

    perplexite = min(perplexite, commentaire_plongement.shape[0] - 1)

    tsne = TSNE(n_components=2, perplexity=perplexite, metric=distance, learning_rate='auto', init='random')

    coord = tsne.fit_transform(commentaire_plongement.loc[:, 'V_000':])

    coord_titre = pd.DataFrame(coord, columns=['x', 'y'], index=commentaire_plongement.index)
    coord_titre['titre'] = commentaire_plongement.titre.values

    coord_titre['review'] = commentaire_plongement.review.apply(
        lambda chaine: '<br>'.join(textwrap.wrap(textwrap.shorten(chaine, width=300), width=50)))

    # coord_titre['utilisateur'] = coord_titre.titre.apply(lambda chaine: 'u' if chaine == LABEL_ENTREE_UTILISATEUR
    # else 'c')

    figure = px.scatter(data_frame=coord_titre, x='x', y='y', color='titre', labels={'x': '', 'y': ''},
                        hover_name='titre', hover_data={'titre': False, 'x': False, 'y': False, 'review': True})

    return figure


def generer_graphique_prediction_film(probabilite: pd.Series) -> go.Figure:
    return go.Figure(data=[go.Bar(x=probabilite.index, y=probabilite)])
