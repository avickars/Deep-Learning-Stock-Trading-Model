import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
from datetime import date, timedelta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
pd.options.mode.chained_assignment = None  # default='warn'
import pulp

def profitPlot(groupData, totalData, startDate, endDate, stocks):
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)
    groupData = pd.DataFrame(groupData, columns=groupData.columns, index=pd.to_datetime(groupData.index))

    # CITATION: https://community.plotly.com/t/plotly-colours-list/11730/2
    DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

    print(groupData)

    fig = go.Figure()
    i = 0
    if 'all' in stocks:
        totalData = pd.DataFrame(totalData, columns=totalData.columns, index=pd.to_datetime(totalData.index))
        filterData = totalData [ (totalData .index >= startDate) & (totalData .index <= endDate)]
        fig.add_trace(
            go.Scatter(x=filterData.index, 
                y=filterData['profit'], 
                name=f"All",
                mode='lines',
                line = dict(color=DEFAULT_PLOTLY_COLORS[i])))
        i = i + 1

    for stock in stocks:
        filterData = groupData[(groupData['stock'] == stock) & (groupData.index >= startDate) & (groupData.index <= endDate)]

        fig.add_trace(
            go.Scatter(x=filterData.index, 
                y=filterData['groupCumProfit'], 
                name=f"{stock}",
                mode='lines',
                line = dict(color=DEFAULT_PLOTLY_COLORS[i])))
        i = i + 1
    return fig

# group = pd.read_csv('group.csv', index_col='date')
# total = pd.read_csv('total.csv', index_col='date')

# profitPlot(group, total, '2018-11-19 00:00:00', '2020-11-16 00:00:00', ['all','ford', 'tesla'])


allStocks = ['ford', 'tesla']
allStocksWithAll = set(allStocks)
allStocksWithAll.add('fdsa')
print(allStocksWithAll)
