import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

stocks = html.Div(
    [
        dbc.Button(
            "Stocks",
            id="stocks-collapse-button",
            className="sidebarbuttons"
        ),
        dbc.Collapse(
            dbc.CardBody(children=[dcc.Dropdown(options=[{"label":"All","value":"all"},
                                                         {"label": "Ford", "value": "ford"},
                                                         {"label": "Tesla", "value": "tesla"},
                                                         {"label": "Honda", "value": "honda"},
                                                         {"label": "General Motors", "value": "gm"},
                                                         {"label": "Fiat/Chrysler", "value": "fiat"}],
                                                id="course-dropdown",
                                                optionHeight=20,
                                                multi=True,
                                                value="All")], id="card-body"),
            id="stocks-collapse"
        ),
    ]
)

# def stocks():
#     # we use this function to make the example items to avoid code duplication
#     return dbc.Card(
#         [
#             dbc.CardHeader(
#                 html.H2(
#                     dbc.Button("Course",
#                                id="stocks-button",
#                                className="sidebarbuttons"),
#                     className="headersidebar"),
#                 className="sidebarcards"
#             ),
#             dbc.Collapse(
#                 # dbc.CardBody(children=[dash_table.DataTable(data=readCourseDataFromApp('canvas/'))], id="card-body"),
#                 dbc.CardBody(children=[dcc.Dropdown(options=[{"label":"Ford","value":"ford"},
#                                                              {"label":"Tesla","value":"tesla"},
#                                                              {"label":"Honda","value":"honda"},
#                                                              {"label":"General Motors","value":"gm"},
#                                                              {"label":"Fiat/Chrysler","value":"fiat"}],
#                                                     id="course-dropdown",
#                                                     optionHeight=120)], id="card-body"),
#                 id="collapse-1"
#             ),
#         ]
#     )
