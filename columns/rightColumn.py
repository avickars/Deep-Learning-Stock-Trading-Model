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

profit = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id = 'profit-plot')
        ]
    ),
    className="mt-3",
)

trades = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(id = 'trade-plot')
        ]
    ),
    className="mt-3",
)


plots = dbc.Tabs(
    [
        dbc.Tab(predictions, label="Predictions"),
        dbc.Tab(profit, label="Profit"),
        dbc.Tab(trades, label="Trades"),
    ]
)

