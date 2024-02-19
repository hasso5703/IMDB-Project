from io import StringIO
from typing import Tuple, Any
from dash import Dash, Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
# import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
# test
from constants_app import *
from comments import filtrer_commentaire

from chart import generer_graphique_projection, generer_graphique_prediction_film
from data_tab import *
from projection_tab import *
from film_prediction_tab import *

"""
===========================================================================
application
"""
application = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

"""
===========================================================================
pied de page
"""
pied_de_page = html.Div(
    dcc.Markdown(
        """
         Cette application a pour but d'analyser des commentaires de films (IMDB) grâce au modèle de langage BERT. 
        """
    ),
    className='p-2 mt-5 bg-primary text-white small'  # padding margin-top
)

"""
===========================================================================
menu Onglets
"""
tabs = dbc.Tabs(
    [
        dbc.Tab([card_donnees], tab_id='Tab_donnees', label='Données'),
        dbc.Tab([card_projection], tab_id='Tab_projection', label='Projection', className='pb-4'),  # padding bottom
        dbc.Tab([card_prediction_film], tab_id='Tab_prediction_film', label='Prédiction film')
    ],
    id='Tabs', active_tab='Tab_projection', className='mt-2'  # margin-top
)

"""
===========================================================================
Mise en page principale (layout)
"""
application.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Commentaires IMDB",
                    className='text-center bg-primary text-white p-2',
                    style={'background-color': 'blue'}
                ),
            )
        ),
        dbc.Row(
            [
                dbc.Col(tabs, width=12, lg=5, className='mt-4 border'),
                dbc.Col(
                    [
                        html.H4("Projection T-SNE"),
                        dcc.Graph(id='Graph_projection', figure={}),
                        html.H4("Prédiction du film"),
                        dcc.Graph(id='Graph_prediction_film', figure={})
                    ],
                    width=12,
                    lg=7,
                    className='pt-4',
                ),
            ],
            className='ms-1',
        ),
        dbc.Row(dbc.Col(pied_de_page)),
        dcc.Store(id='Store_donnees_partage', storage_type='session', data={}),
        dcc.Store(id='Donnees_knn', storage_type='session', data={})
    ],
    fluid=True,
)

"""
===========================================================================
callback
"""


@application.callback(
    Output('Store_donnees_partage', 'data'),
    Input('Button_predire_film', 'n_clicks'),
    State('Textarea_commentaire', 'value'),
    State('liste_deroulante', 'value')
)
def stocker_commentaire_utilisateur(n_click: int, commentaire: str, liste_films: list[str]) -> dict:
    if n_click is None or n_click == 0 or commentaire is None or len(commentaire.strip()) == 0 or len(liste_films) < 2:
        return {}
    plongement = MODELE_LANGAGE(commentaire.strip())
    return {'commentaire': commentaire, 'plongement': plongement.to_json(orient='split'),
            'films_selectionnees': liste_films}  # il faut que l'on puisse générer un Json


@application.callback(
    Output('Graph_projection', 'figure'),
    Output('Donnees_knn', 'data'),
    Input('Slider_nombre_commentaire', 'value'),
    Input('RangeSlider_nombre_mot', 'value'),
    Input('Slider_perplexite', 'value'),
    Input('Dropdown_distance', 'value'),
    Input('Store_donnees_partage', 'data'),
    State('RadioItems_methode_prediction_film', 'value')
)
def mettre_a_jour_figure_projection(nb_max_commentaire_par_film: int, min_max_mot: tuple[int, int],
                                    perplexite: int, distance: str,
                                    donnees_partage: dict,
                                    selected_method: str) -> tuple[Any, Any]:
    df_filtre = filtrer_commentaire(DF_COMMENTAIRE_PLONGEMENT,
                                    nb_max_commentaire_par_film=nb_max_commentaire_par_film,
                                    min_max_mot=min_max_mot)
    films_selectionnes = donnees_partage.get("films_selectionnees", [])
    if films_selectionnes:  # Vérifier si des films sont sélectionnés
        df_filtre = df_filtre.query("titre in @films_selectionnes")
        # df_filtre[df_filtre['titre'].isin(films_selectionnes)]
    else:
        df_filtre = df_filtre.query("titre in @LISTE_FILMS_INITIAL")
        # df_filtre[df_filtre['titre'].isin(LISTE_FILMS_INITIAL)]

    if selected_method == "KNN (K-PPV)":
        print("df filtres", df_filtre)
        all_embeddings = df_filtre.loc[:, 'V_000':].values
        labels = df_filtre.titre.values
        k = 5
        knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_model.fit(all_embeddings, labels)
        df_plongement = pd.read_json(StringIO(donnees_partage['plongement']), orient='split')
        nouveau_plongement = df_plongement.loc[:, 'V_000':].values
        probabilites = knn_model.predict_proba(nouveau_plongement)
        classes = knn_model.classes_
        print(classes)
        print(probabilites)
        donnees_knn = {'probabilites': probabilites, 'modalites': classes}
    else:
        donnees_knn = {}

    if donnees_partage:
        df_plongement = pd.read_json(StringIO(donnees_partage['plongement']), orient='split')
        df_une_ligne = pd.DataFrame(
            {'titre': [LABEL_COMMENTAIRE_UTILISATEUR], 'rating': [-1], 'review': [donnees_partage['commentaire']]})
        df_une_ligne = pd.concat([df_une_ligne, df_plongement], axis=1)
        df_filtre = pd.concat([df_filtre, df_une_ligne])

    figure = generer_graphique_projection(df_filtre, perplexite=perplexite, distance=distance,
                                          selected_method=selected_method)

    return figure, donnees_knn


@application.callback(
    Output('Graph_prediction_film', 'figure'),
    State('Store_donnees_partage', 'data'),
    Input('Donnees_knn', 'data'),
    State('RadioItems_methode_prediction_film', 'value')
)
def mettre_a_jour_figure_prediction(donnees_partage: dict, donnees_knn: dict, selected_method: str) -> go.Figure:
    if donnees_partage:
        if selected_method == 'réseau de neurones':
            plongement = pd.read_json(StringIO(donnees_partage['plongement']), orient='split').to_numpy()
            probabilite = RESEAU_NEURONE.predict(plongement)[0]
            dictionnaire_films = {nom_film: proba for nom_film, proba in zip(LISTE_FILMS, probabilite)
                                  if nom_film in donnees_partage.get('films_selectionnees', [])}
            modalite = list(dictionnaire_films.keys())
            probabilite = list(dictionnaire_films.values())
            somme_probabilites = sum(probabilite)
            probabilite = [prob / somme_probabilites for prob in probabilite]
        else:
            if donnees_knn:
                probabilite = donnees_knn.get('probabilites', [])[0]
                modalite = donnees_knn.get('modalites', [])
    else:
        NB_TITRE = len(LISTE_FILMS_INITIAL)
        probabilite = [1 / NB_TITRE for i in range(NB_TITRE)]
        modalite = LISTE_FILMS_INITIAL

    figure = generer_graphique_prediction_film(pd.Series(probabilite, index=modalite))
    return figure


@application.callback(
    Output('alert', 'is_open'),
    Input('liste_deroulante', 'value')
)
def toggle_alert(selected_films):
    return selected_films is None or len(selected_films) < 2


@application.callback(
    Output('alert-text-empty', 'is_open'),
    Input('Button_predire_film', 'n_clicks'),
    State('Textarea_commentaire', 'value')
)
def toggle_alert(n_clicks, input_text):
    return n_clicks and (input_text is None or input_text.strip() == '')


"""
===========================================================================
lancement du serveur
"""
if __name__ == '__main__':
    application.run_server(debug=True)
