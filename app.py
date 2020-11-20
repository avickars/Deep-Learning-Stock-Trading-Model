import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from columns.leftColumn import stocks, timeLine, seed, date
from columns.rightColumn import plots
from dash.dependencies import Input, Output, State
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Reading in Data
ford = pd.read_csv('Data/fordComplete.csv', index_col = 'Date')

navBar = dbc.Navbar(children=[
    html.A(
        # Use row and col to control vertical alignment of logo / brand
        dbc.Row(
            [
                html.Img(src="assets/CMPTLogo.png"),
                dbc.Col(dbc.NavbarBrand("CMPT 353 Project: Auto Stock Predictor", className="ml-2"), align="end"),
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
        html.Div(html.Div(children=[
            html.Br(),
            stocks,
            html.Br(),
            timeLine,
            html.Br(),
            seed
                                    ], id="inner-column-left", className="column"), className="col-outer", id="outer-column-left"),
        html.Div(html.Div(children=[
            plots

        ], id="inner-column-right"), className="col-outer", id="outer-column-right")
    ])
], id="base")

# Left column collapse
@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in range(1, 4)],
    [Input(f"group-{i}-toggle", "n_clicks") for i in range(1, 4)],
    [State(f"collapse-{i}", "is_open") for i in range(1, 4)],
)
def toggle_accordion(n1, n2, n3, is_open1, is_open2, is_open3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "group-1-toggle" and n1:
        return not is_open1, False, False
    elif button_id == "group-2-toggle" and n2:
        return False, not is_open2, False
    elif button_id == "group-3-toggle" and n3:
        return False, False, not is_open3
    return False, False, False

@app.callback(
    dash.dependencies.Output('output-container-range-slider', 'children'),
    [dash.dependencies.Input('my-range-slider', 'value')])
def update_output(value):
    print(f"{str(date.iloc[value[0],0])} - {str(date.iloc[value[1],0])}")
    return f"{str(date.iloc[value[0],0])[0:10]} - {str(date.iloc[value[1],0])[0:10]}"
    


# Start the Dash server
if __name__ == '__main__':
    app.run_server()
