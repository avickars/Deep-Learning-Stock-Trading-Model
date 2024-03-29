import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from columns.leftColumn import stocks, timeLine, seed, date, submit, risk
from columns.rightColumn import predictions, profit, trades
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
from project_functions.dashboard_functions import trader, predictionPlot, profitPlot, tradePlot
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Reading in Data
data = pd.read_csv('Data/Final Predictions/dataComplete.csv', index_col = 'Date')

date = pd.read_csv('Data/Final Predictions/dateRange.csv')
date['Date'] = pd.to_datetime(date['Date'])

navBar = dbc.Navbar(children=[
    html.A(
        # Use row and col to control vertical alignment of logo / brand
        dbc.Row(
            [
                html.Img(src="assets/CMPTLogo.png"),
                dbc.Col(dbc.NavbarBrand("CMPT 353 Project: Stock Predictor", className="ml-2"), align="end"),
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
            seed,
            html.Br(),
            risk,
            html.Br(),
            submit], id="inner-column-left", className="column"), className="col-outer", id="outer-column-left"),
        html.Div(html.Div(children=[
            dcc.Tabs([
                dcc.Tab(label="Predictions", children=[predictions], className = 'os-tab'),
                dcc.Tab(label="Trades", children=[trades], className = 'os-tab'),
                dcc.Tab(label="Profits", children=[profit], className = 'os-tab')
            ], className="os-tab-container", id="tabs")
        ], id="inner-column-right"), className="col-outer", id="outer-column-right")
        # html.Div(html.Div(children=[plots], id="inner-column-right"), className="col-outer", id="outer-column-right")
    ])
], id="base")

# Left column collapse
@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in range(1, 6)],
    [Input(f"group-{i}-toggle", "n_clicks") for i in range(1, 6)],
    [State(f"collapse-{i}", "is_open") for i in range(1, 6)],
)
def toggle_accordion(n1, n2, n3, n4, n5, is_open1, is_open2, is_open3, is_open4, is_open5):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "group-1-toggle" and n1:
        return not is_open1, False, False, False, False
    elif button_id == "group-2-toggle" and n2:
        return False, not is_open2, False, False, False
    elif button_id == "group-3-toggle" and n3:
        return False, False, not is_open3, False, False
    elif button_id == "group-4-toggle" and n4:
        return False, False, False, not is_open4, False
    elif button_id == "group-5-toggle" and n5:
        return False, False, False, False, not is_open5
    return False, False, False, False, False

# Call back for date slider in left column
@app.callback(
    Output('output-container-range-slider', 'children'),
    [Input('my-range-slider', 'value')])
def update_output(value):
    return f"{str(date.iloc[value[0],0])[0:10]} - {str(date.iloc[value[1],0])[0:10]}"

# Callback for submit button
@app.callback([
    Output('prediction-plot', 'figure'),
    Output('profit-plot', 'figure'),
    Output('trade-plot', 'figure')
], [Input("submit-changes-button", 'n_clicks')],
state=[
    State("course-dropdown", 'value'),
    State('my-range-slider', 'value'),
    State("seed-input", 'value'),
    State("risk-input", 'value')]
)
def filterDashboard(numClicks, stocks, dateRange, seedValue, riskInput):
    # return [predictionPlot(data, date.iloc[dateRange[0],0], date.iloc[dateRange[1],0],["ford"])]
    if numClicks is None:  # Default Option
        groupProfits, totalProfits, stockTrades = trader(date.iloc[dateRange[0],0],  date.iloc[dateRange[1],0], seedValue, data, stocks, riskInput)
        return [predictionPlot(data, date.iloc[dateRange[0],0], date.iloc[dateRange[1],0],stocks),
                profitPlot(groupProfits, totalProfits, date.iloc[dateRange[0],0], date.iloc[dateRange[1],0], stocks),
                tradePlot(stockTrades, date.iloc[dateRange[0],0],  date.iloc[dateRange[1],0],stocks)]
    else:
        groupProfits, totalProfits, stockTrades = trader(date.iloc[dateRange[0],0],  date.iloc[dateRange[1],0], seedValue, data, stocks, riskInput)
        return [predictionPlot(data, date.iloc[dateRange[0],0], date.iloc[dateRange[1],0],stocks),
                profitPlot(groupProfits, totalProfits, date.iloc[dateRange[0],0], date.iloc[dateRange[1],0], stocks),
                tradePlot(stockTrades, date.iloc[dateRange[0],0],  date.iloc[dateRange[1],0],stocks)]

# Start the Dash server
if __name__ == '__main__':
    app.run_server()
