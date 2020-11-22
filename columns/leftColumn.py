import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

date = pd.read_csv('Data/dateRange.csv')
date['Date'] = pd.to_datetime(date['Date'])


stocks = html.Div(
    [
        dbc.Button(
            "Stocks",
            id="group-1-toggle",
            className="sidebarbuttons"
        ),
        dbc.Collapse(
            dbc.CardBody(children=[dcc.Dropdown(options=[{"label": "All", "value": "all"},
                                                         {"label": "Ford", "value": "ford"},
                                                         {"label": "Tesla", "value": "tesla"},
                                                         {"label": "Honda", "value": "honda"},
                                                         {"label": "General Motors", "value": "gm"},
                                                         {"label": "Fiat/Chrysler", "value": "fiat"}],
                                                id="course-dropdown",
                                                optionHeight=20,
                                                multi=True,
                                                value="all")], className='collapse-body'),
            id="collapse-1"
        ),
    ]
)

timeLine = html.Div(
    [
        dbc.Button(
            "Time Range",
            id="group-2-toggle",
            className="sidebarbuttons"
        ),
        dbc.Collapse(
            dbc.CardBody(children=[dcc.RangeSlider(
                id='my-range-slider',
                min=0,
                max=len(date)-1,
                step=1,
                value=[0, len(date)-1]
            ),
                html.Div(id='output-container-range-slider')], className='collapse-body', id="timeLine-collapse"),
            id="collapse-2"
        ),
    ]
)


seed = html.Div(
    [
        dbc.Button(
            "Seed Money",
            id="group-3-toggle",
            className="sidebarbuttons"
        ),
        dbc.Collapse(
            dbc.CardBody(children=[
                dcc.Input(
                    id="seed-input", type="number", min=10, step=10, value = 10
                )
            ], className='collapse-body', id="seed-collapse"),
            id="collapse-3"
        ),
    ]
)

submit = html.Div(
    [
        dbc.Button(
            "Submit Changes",
            id="group-4-toggle",
            className="sidebarbuttons"
        ),
        dbc.Collapse(
            dbc.CardBody(children=[
                 dbc.Button(
                    "Submit",
                    id="submit-changes-button"
        )
            ], className='collapse-body', id="submit-collapse"),
            id="collapse-4"
        ),
    ]
)

risk = html.Div(
    [
        dbc.Button(
            "Risk",
            id="group-5-toggle",
            className="sidebarbuttons"
        ),
        dbc.Collapse(
            dbc.CardBody(children=[
                dcc.Input(
                    id="risk-input", type="number", min=0, max=1, step=0.05, value = 0.5
                )
            ], className='collapse-body', id="risk-collapse"),
            id="collapse-5"
        ),
    ]
)


