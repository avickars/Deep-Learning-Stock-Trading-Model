import dash_bootstrap_components as dbc
import dash_core_components as dcc

predictions = dcc.Graph(id = 'prediction-plot',style = {'layout.autosize':'true','width': '100%', 'height':'850px'})

profit = dcc.Graph(id = 'profit-plot',style = {'layout.autosize':'true','width': '100%', 'height':'850px'})

trades = dcc.Graph(id = 'trade-plot',  style = {'layout.autosize':'true','width': '100%', 'height':'850px'})


