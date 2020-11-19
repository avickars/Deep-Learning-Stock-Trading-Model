import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

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
                                                value="All")], className='collapse-body'),
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
                max=20,
                step=0.5,
                value=[5, 15]
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
                    id="input_range", type="number", min=10, step=10, value = 10
                )
            ], className='collapse-body', id="seed-collapse"),
            id="collapse-3"
        ),
    ]
)
