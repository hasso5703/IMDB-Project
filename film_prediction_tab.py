from dash import dcc, html
import dash_bootstrap_components as dbc
from constants_app import LISTE_FILMS_INITIAL, LISTE_FILMS, K

"""
===========================================================================
onglet 'Classification'
"""
card_prediction_film = dbc.Card(
    [
        html.H4("Méthode de prédiction:", className='card-title'),
        dcc.RadioItems(id='RadioItems_methode_prediction_film',
                       options=[{'label': " " + methode, 'value': methode} for methode in
                                ['réseau de neurones', 'KNN (K-PPV)']],
                       value='réseau de neurones'),
        html.H5("paramètre K:", className='card-title'),
        dcc.Slider(id='Slider_k',
                   marks={i: f'{i}' for i in range(1, 20, 1)},
                   min=1,
                   max=20,
                   step=1,
                   value=K,
                   included=False),
        html.H4("Liste des films:", className='card-title'),
        dcc.Dropdown(
            id='liste_deroulante',
            options=LISTE_FILMS,
            value=LISTE_FILMS_INITIAL,
            multi=True
        ),
        dbc.Alert("Veuillez sélectionner au moins deux films.", id='alert', color="danger", dismissable=True,
                  is_open=True),
        html.H4("Nouveau commentaire:", className='card-title mt-3'),
        dbc.Textarea(id='Textarea_commentaire'),
        dbc.Button(id='Button_predire_film', children='Prédire le film', color='primary', className='m-2'),
        dbc.Alert("Veuillez entrer du texte dans le champ.", id='alert-text-empty', color="danger", dismissable=True,
                  is_open=False)
    ],
    body=True,
    className='mt-4'
)
