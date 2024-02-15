from dash import dcc, html, Input, Output, Dash

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Couleur'),
    html.H2('Choisir une couleur'),
    dcc.Dropdown(id='Dropdown_couleur', value='Rouge',
                 options=[{'label': couleur, 'value': couleur}
                          for couleur in ['Rouge', 'Vert', 'Bleu']]),
    html.Div(id='Div_afficher_couleur', children=[])
])


@app.callback(Output('Div_afficher_couleur', 'children'),
              Input('Dropdown_couleur', 'value'))
def afficher_couleur(couleur: str) -> str:
    if couleur is None:
        return []

    return [html.H3(couleur + " a été choisie")]


if __name__ == '__main__':
    app.run_server(debug=True)