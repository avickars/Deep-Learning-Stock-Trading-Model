import pandas as pd
from datetime import date, timedelta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
pd.options.mode.chained_assignment = None  # default='warn'

def trader(startDate, endDate, seedMoney, data, selectedStocks, risk):
    # Ensuring the start/end dates are in the correct format
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)

    # Ensuring the df indexes are in proper format
    data = pd.DataFrame(data, columns=data.columns, index=pd.to_datetime(data.index))

    # Filter for dates we dont want
    data = data[data.index >= startDate]
    data = data[data.index <= endDate]


    stocks = {}

    for stock in selectedStocks:
        stocks[stock] = data[data['stock'] == stock]

    # Initialize df to hold all the stocks bought
    stockTrades = pd.DataFrame(columns = ['date', 'stock', 'buyPrice', 'numShares', 'sellPrice', 'sellDate', 'profit'])
    
    # Initialize variable to hold the cash we currently have to spend
    liquidity = seedMoney

    i = 1
    for date in stocks[selectedStocks[0]].index: # Ierating through each day
        for stock in selectedStocks: # Iterating through each stock to make sure we sell off any stock that we want to

            # Lets sell some stock!
            if stocks[stock].loc[date,:]['predictedAction'] == 'sell':
                if len(stockTrades[stockTrades.isnull().any(axis=1)]) > 0: # Testing if we own any stocks of this company, if yes continue and sell them
                    # Recording the price the stock is sold at
                    stockTrades.loc[stockTrades[stockTrades.isnull().any(axis=1)].index.values,'sellPrice'] = stocks[stock].loc[date,:]['Open']

           
                    # Getting the indexes of stocks we need to sell (i.e. rows that contain a null value since they have no)
                    indices = stockTrades[stockTrades.isnull().any(axis=1)].index.values

                    # Recording the profit made on this trade
                    moneyOut = np.round(stockTrades.loc[indices, 'numShares'].values[0] * stockTrades.loc[indices, 'buyPrice'].values[0], decimals=2)
                    moneyIn = np.round(stockTrades.loc[indices, 'numShares'].values[0] * stockTrades.loc[indices, 'sellPrice'].values[0], decimals=2)
                    stockTrades.loc[indices,'profit'] = stocks[stock].loc[date,:]['Open'] = moneyIn - moneyOut
                    stockTrades.loc[indices,'sellDate'] = date
                    
                    # Updating the liquidity value to reflect the stocks we just sold
                    liquidity = liquidity + moneyIn

        for stock in selectedStocks: # Iterating through each stock to buy any stock that we want to

            # Lets buy some stock!
            if stocks[stock].loc[date,:]['predictedAction'] == 'buy': # Testing if we even want to buy this stock today
                price = stocks[stock].loc[date,:]['Open']
                numShares = np.floor(liquidity/price)
                
                # if numShares > 0: # Testing if we have enough liquidity to buy any shares, if yes continue otherwise skip
                #     liquidity = liquidity - np.round(numShares * price,decimals=2)
                #     stockTrades = stockTrades.append({'date':date, 'stock':stock, 'buyPrice': price, 'numShares': numShares}, ignore_index=True)


    # Creating a running sum of profit
    stockTrades['cumProfit'] = stockTrades['profit'].cumsum()

    return stockTrades

def predictionPlot(data, startDate, endDate, stocks):
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)
    data = pd.DataFrame(data, columns=data.columns, index=pd.to_datetime(data.index))



    # CITATION: https://community.plotly.com/t/plotly-colours-list/11730/2
    DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

    fig = go.Figure()
    i = 0
    for stock in stocks:
        filterData = data[(data['stock'] == stock) & (data.index >= startDate) & (data.index <= endDate)]

        fig.add_trace(
            go.Scatter(x=filterData.index, 
                y=filterData['Open'], 
                name=f"{stock} - Truth",
                mode='lines',
                line = dict(color=DEFAULT_PLOTLY_COLORS[i])))

        fig.add_trace(
            go.Scatter(x=filterData.index, 
                y=filterData['predicted'],
                name=f"{stock} - Predicted",
                line = dict(dash='dot', width=3, color=DEFAULT_PLOTLY_COLORS[i])))
        i = i + 1\

    return fig


# Reading in Data
data = pd.read_csv('Data/dataComplete.csv', index_col = 'Date')
print(trader('2018-11-19 00:00:00', '2020-11-16 00:00:00', 1000, data, ['ford', 'tesla'], 0.5))



