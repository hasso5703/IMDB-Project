from dash import dcc, html
import dash_bootstrap_components as dbc
from constants import MIN_MAX_MOT_INITIAL, NB_MAX_COMMENTAIRE_PAR_FILM_INITIAL

"""
===========================================================================
onglet 'Données'
"""
card_donnees = dbc.Card(
    [
        html.H4("Nombre minimum et maximum de mots:", className='card-title mt-3'),
        dcc.RangeSlider(
            id='RangeSlider_nombre_mot',
            marks={i: f'{i}' for i in range(100, 360, 20)},
            min=100,
            max=340,
            step=20,
            value=MIN_MAX_MOT_INITIAL,
            included=True
        ),
        html.H4("Nombre de commentaires par film:", className='card-title'),
        dcc.Slider(
            id='Slider_nombre_commentaire',
            marks={i: f'{i}' for i in range(5, 105, 10)},
            min=5,
            max=100,
            step=10,
            value=NB_MAX_COMMENTAIRE_PAR_FILM_INITIAL,
            included=False,
        )
    ],
    body=True,
    className='mt-4',
)



