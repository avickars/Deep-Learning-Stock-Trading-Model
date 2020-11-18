import dash_bootstrap_components as dbc
import dash_html_components as html

timeSeriesPlot = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 1!", className="card-text"),
            dbc.Button("Click here", color="success"),
        ]
    ),
    className="mt-3",
)

MSE = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
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
        dbc.Tab(timeSeriesPlot, label="Predictions"),
        dbc.Tab(MSE, label="Mean Squared Error"),
        dbc.Tab(resids, label="Residuals"),
    ]
)

