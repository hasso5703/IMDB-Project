from datetime import datetime as dt  # Importez datetime et renommez-le dt

from dash import dcc, html
import dash_bootstrap_components as dbc

from constants import *

"""
===========================================================================
onglet 'Classification'
"""
card_temporalite = dbc.Card(
    [
        html.H4("Précision:", className='card-title'),
        dcc.RadioItems(
            id='RadioItems_temporalite',
            options=[{'label': " " + methode,  'value': methode} for methode in ['années', 'mois']],
            value='années',
            inline=True
        ),
        html.H4("Sélectionnez la plage de dates :", className='card-title mt-3'),
        dcc.DatePickerRange(
            id='DateRangePicker_dates_bornes',
            min_date_allowed=dt(2010, 1, 1),
            max_date_allowed=dt.now(),        # Date maximale autorisée (maintenant)
            initial_visible_month=dt.now(),
            start_date=dt(2010, 1, 1),
            end_date=dt.now()
        )
    ],
    body=True,
    className='mt-4'
)