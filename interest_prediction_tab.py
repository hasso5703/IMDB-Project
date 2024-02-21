from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from constants_app import conf_matrix, TPR, FPR
import numpy as np

"""
===========================================================================
onglet 'Prédiction intérêt'
"""


def style_cell(value):
    if value >= 10000:
        return {'backgroundColor': 'green', 'color': 'white'}
    else:
        return {'backgroundColor': 'white', 'color': 'black'}


card_interest_prediction = dbc.Card(
    html.Div([
        html.H3('Intérêt :'),
        dbc.Button(id='button_interet', children="Prédire"),
        dcc.Graph(id='Graph_interet_film', figure={}),
        html.Div([
            html.H3('Taux de bonne prédiction: {:.2f}%'.format(0.85 * 100)),
            html.H3('Matrice de confusion:'),
            html.Table(
                # Lignes de la table
                children=[
                    # Ligne 1
                    html.Tr(
                        # Colonnes de la ligne
                        children=[
                            # Cellule 1
                            html.Td(conf_matrix[0][0], style=style_cell(conf_matrix[0][0])),
                            # Cellule 2
                            html.Td(conf_matrix[0][1], style=style_cell(conf_matrix[0][1]))
                        ]
                    ),
                    # Ligne 2
                    html.Tr(
                        # Colonnes de la ligne
                        children=[
                            # Cellule 3
                            html.Td(conf_matrix[1][0], style=style_cell(conf_matrix[1][0])),
                            # Cellule 4
                            html.Td(conf_matrix[1][1], style=style_cell(conf_matrix[1][1]))
                        ]
                    )
                ]
            ),
            html.H3('Courbe ROC:'),
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=[FPR], y=[TPR], mode='lines', name='ROC Curve')],
                    layout=go.Layout(title='ROC Curve')
                )
            )
        ])
    ]),
    id='output-probability-card',  # enveloppez l'ID dans une liste
    body=True,
    className='mt-4',
)
