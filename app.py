import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from columns.leftColumn import stocks


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

navBar = dbc.Navbar(children=[
    html.A(
        # Use row and col to control vertical alignment of logo / brand
        dbc.Row(
            [
                html.Img(src="assets/CMPTLogo.png"),
                dbc.Col(dbc.NavbarBrand("CMPT 353 Project: Auto Stock Predictor", className="ml-2"),align="end"),
            ],
            no_gutters=True,
        ),
    )
],
    color="darkred",
    dark=True,
)


# App layout
app.layout = html.Div(children=[
    html.Div(id="row-1", children=[navBar]),
    html.Div(id="row-2", children=[
        html.Div(html.Div(children=[stocks()
        ], id="inner-column-left", className="column"), className="col-outer", id="outer-column-left"),
        html.Div(html.Div(children=[

        ], id="inner-column-right"), className="col-outer", id="outer-column-right")
    ])
], id="base")

# Start the Dash server
if __name__ == '__main__':
    app.run_server()
