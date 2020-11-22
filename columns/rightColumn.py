import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import dash_core_components as dcc
# from functions import trader



predictions = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id = 'prediction-plot')
        ]
    ),
    className="mt-3",
)

MSE = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id = 'profit-plot')
        ]
    ),
    className="mt-3",
)

resids = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)


plots = dbc.Tabs(
    [
        dbc.Tab(predictions, label="Predictions"),
        dbc.Tab(MSE, label="Profit"),
        dbc.Tab(resids, label="Residuals"),
    ]
)

