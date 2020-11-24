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
            dbc.CardBody(children=[dcc.Dropdown(options=[{"label": "Ford", "value": "ford"},
                                                         {"label": "Nordstrom", "value": "nordstrom"},
                                                         {"label": "Bank of America", "value": "boa"},
                                                         {"label": "Exxon Mobil", "value": "exxon"},
                                                         {"label": "Forward Industries", "value": "forward"}],
                                                id="course-dropdown",
                                                optionHeight=20,
                                                multi=True,
                                                value=['ford','nordstrom','boa','exxon','forward'])], className='collapse-body'),
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
                    id="seed-input", type="number", min=1000, step=100, value = 1000
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
                    id="submit-changes-button",
                    className='submit-button'
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
                    id="risk-input", type="number", min=0, max=1, step=0.05, value = 1
                )
            ], className='collapse-body', id="risk-collapse"),
            id="collapse-5"
        ),
    ]
)


