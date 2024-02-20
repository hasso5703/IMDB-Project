import sklearn
from dash import dcc, html
import dash_bootstrap_components as dbc
from constants import PERPLEXITE_TSNE_INITIAL, DISTANCE_TSNE_INITIAL

"""
===========================================================================
onglet 'Projection'
"""
card_projection = dbc.Card(
    [
        html.H4("Choix de la méthode:", className='card-title'),
        dcc.Dropdown(
            id='Dropdown_method_reduction',
            options=[
                {'label': 'Analyse en Composantes Principales (ACP)', 'value': 'ACP'},
                {'label': 't-Distributed Stochastic Neighbor Embedding (t-SNE)', 'value': 't-SNE'},
                {'label': 'Multidimensionnal scaling', 'value': 'Mult'}
            ],
            value='t-SNE',
            clearable=False
        ),
        html.H4("Perplexité de la méthode T-SNE:", className='card-title'),
        dcc.Slider(
            id='Slider_perplexite',
            marks={i: f'{i}' for i in range(5, 70, 5)},
            min=5,
            max=60,
            step=5,
            value=PERPLEXITE_TSNE_INITIAL,
            included=False
        ),
        html.H4("Distance de la méthode T-SNE:", className='card-title mt-3'),
        dcc.Dropdown(
            id='Dropdown_distance',
            multi=False,
            options=[{'label': d, 'value': d} for d in sklearn.metrics.pairwise.distance_metrics()],
            value=DISTANCE_TSNE_INITIAL,
            clearable=False
        )
    ],
    body=True,
    className='mt-4',
)
