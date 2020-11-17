import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

def stocks():
    # we use this function to make the example items to avoid code duplication
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H2(
                    dbc.Button("Course",
                               id="group-1-toggle",
                               className="sidebarbuttons"),
                    className="headersidebar"),
                className="sidebarcards"
            ),
            dbc.Collapse(
                # dbc.CardBody(children=[dash_table.DataTable(data=readCourseDataFromApp('canvas/'))], id="card-body"),
                dbc.CardBody(children=[dcc.Dropdown(options=[{"label":"Ford","value":"ford"},
                                                             {"label":"Tesla","value":"tesla"},
                                                             {"label":"Honda","value":"honda"},
                                                             {"label":"General Motors","value":"gm"},
                                                             {"label":"Fiat/Chrysler","value":"fiat"}],
                                                    id="course-dropdown",
                                                    optionHeight=120)], id="card-body"),
                id="collapse-1"
            ),
        ]
    )