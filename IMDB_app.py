from io import StringIO
from typing import Tuple, Any
from dash import Dash, Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
# import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# test
from constants_app import *
from comments import filtrer_commentaire

from chart import generer_graphique_projection, generer_graphique_prediction_film
from data_tab import *
from projection_tab import *
from film_prediction_tab import *
from interest_prediction_tab import *
from tab_temporalite import *

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
        dbc.Tab([card_prediction_film], tab_id='Tab_prediction_film', label='Prédiction film'),
        dbc.Tab([card_interest_prediction], tab_id='Tab_interest_prediction', label='Prédiction intérêt'),
        dbc.Tab([card_temporalite], tab_id='Tab_temporalite', label='Temporalité')
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
                        dcc.Graph(id='Graph_prediction_film', figure={}),
                        html.H4("Evolution commentaires"),
                        dcc.Graph(id='Graph_evolution', figure={}),
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
    State('RadioItems_methode_prediction_film', 'value'),
    State("Slider_k", "value")
)
def mettre_a_jour_figure_projection(nb_max_commentaire_par_film: int, min_max_mot: tuple[int, int],
                                    perplexite: int, distance: str,
                                    donnees_partage: dict,
                                    selected_method: str, k: int) -> tuple[Any, Any]:
    df_filtre = filtrer_commentaire(DF_COMMENTAIRE_PLONGEMENT,
                                    nb_max_commentaire_par_film=nb_max_commentaire_par_film,
                                    min_max_mot=min_max_mot)
    films_selectionnes = donnees_partage.get("films_selectionnees", [])
    if films_selectionnes:
        df_filtre = df_filtre.query("titre in @films_selectionnes")
    else:
        df_filtre = df_filtre.query("titre in @LISTE_FILMS_INITIAL")

    if selected_method == "KNN (K-PPV)" and films_selectionnes:
        all_embeddings = df_filtre.loc[:, 'V_000':].values
        labels = df_filtre.titre.values
        knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_model.fit(all_embeddings, labels)
        df_plongement = pd.read_json(StringIO(donnees_partage['plongement']), orient='split')
        nouveau_plongement = df_plongement.loc[:, 'V_000':].values
        probabilites = knn_model.predict_proba(nouveau_plongement)
        classes = knn_model.classes_
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
            probabilite = donnees_knn.get('probabilites', [])[0]
            modalite = donnees_knn.get('modalites', [])
    else:
        NB_TITRE = len(LISTE_FILMS_INITIAL)
        probabilite = [1 / NB_TITRE for i in range(NB_TITRE)]
        modalite = LISTE_FILMS_INITIAL

    figure = generer_graphique_prediction_film(pd.Series(probabilite, index=modalite))
    return figure


# Callback pour effectuer la prédiction lorsque l'utilisateur soumet un commentaire
@application.callback(
    Output('Graph_interet_film', 'figure'),
    State('Store_donnees_partage', 'data'),
    Input("button_interet", "n_clicks")
)
def prediction_interet(donnees_partage: dict, n_clicks: int):
    if donnees_partage and n_clicks:
        plongement = pd.read_json(StringIO(donnees_partage['plongement']), orient='split').to_numpy()
        prediction = RESEAU_NEURONE_RATING.predict(plongement)[0]
        print(prediction)
        interet = "oui" if max(prediction) == prediction[1] else "non"
        prediction_str = "intérêt : " + str(interet)
    else:
        prediction_str = "intérêt : ..."
        prediction = [0, 0]
    figure = generer_graphique_prediction_film(pd.Series(prediction, index=["non", "oui"]))

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
@application.callback(
    Output('Graph_evolution', 'figure'),
    Input('RadioItems_methode_prediction_film', 'n_clicks'),
    State('RadioItems_methode_prediction_film', 'value'),
    State('DateRangePicker_dates_bornes', 'start_date'),
    State('DateRangePicker_dates_bornes', 'end_date')
)
def mettre_a_jour_figure_evolution(donnees_partage: dict, methode_selectionnee: str, start_date: str,
                                   end_date: str) -> go.Figure:
    films = LISTE_FILMS
    annees_uniques = []
    mois_et_annees_uniques = []
    nombres_commentaires = []

    for film in films:
        df = pd.read_csv(chemin_fichier)

        # Convertir la colonne 'date' en datetime en spécifiant le format littéral
        df['date'] = pd.to_datetime(df['date'], format='%d %B %Y')
        annees_uniques.extend(df['date'].dt.year.unique().tolist())
        # Grouper les revues par année:mois
        if methode_selectionnee == "années":
            reviews = df.groupby(df['date'].dt.year).size().tolist()
        elif methode_selectionnee == "mois":
            reviews = df.groupby(df['date'].dt.month).size().tolist()
        nombres_commentaires.append(reviews)
        mois_et_annees = df['date'].dt.strftime('%m-%Y').unique().tolist()
        mois_et_annees_uniques.extend(mois_et_annees)

    print("Nombre de commentaires pour chaque film par année :", nombres_commentaires)
    annees_uniques = sorted(list(set(annees_uniques)))
    mois_et_annees_uniques = sorted(list(set(mois_et_annees_uniques)))
    if methode_selectionnee == "années":
        temps = annees_uniques
    elif methode_selectionnee == "mois":
        temps = mois_et_annees_uniques
    # Création des traces pour chaque film
    data = []
    for i, film in enumerate(films):
        trace = go.Scatter(
            # à changer de mois ou année selon le bouton
            x=temps,
            y=nombres_commentaires[i],
            mode='lines+markers',
            name=film
        )
        data.append(trace)

    # Création du layout
    layout = go.Layout(
        title='Evolution du nombre de commentaires par film',
        xaxis=dict(title='Temps'),
        yaxis=dict(title='Nombre de commentaires')
    )

    # Création de la figure
    figure = go.Figure(data=data, layout=layout)

    return figure
"""

"""
===========================================================================
lancement du serveur
"""
if __name__ == '__main__':
    application.run_server(debug=True)
