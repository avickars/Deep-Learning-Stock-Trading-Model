import pandas as pd
from datetime import date, timedelta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
pd.options.mode.chained_assignment = None  # default='warn'
import pulp

def optimizer(data, liquidity, risk):
    data = data.set_index('stock')

    # CITATION: https://towardsdatascience.com/linear-programming-and-discrete-optimization-with-python-using-pulp-449f3c5f6e99
    # CITATION: https://towardsdatascience.com/linear-programming-using-python-priyansh-22b5ee888fe0
    
    # Defining the IP
    stockOptimizer = pulp.LpProblem("Maximize Profits",pulp.LpMaximize)

    # Defining the variables
    variables = [stock for stock in data.index]
    ipVariables = pulp.LpVariable.matrix("",variables, cat = "Integer", lowBound=0)

    # Defining objective function
    stockOptimizer += pulp.lpSum(ipVariables * data['expectedProfit'].values), "Z"

    # Defining the constrain that we don't spend more money than our liquidity
    stockOptimizer += pulp.lpSum(ipVariables * data['price'].values) <= liquidity

    # Defining constrains that ensure we don't hold any one stock more than our risk amount allows
    for stock in ipVariables:
        stockOptimizer += pulp.lpSum(stock * data.loc[stock.name[1:],'price']) <= liquidity * risk

    # Solving the IP
    stockOptimizer.solve(pulp.PULP_CBC_CMD(msg=0))

    for variable in stockOptimizer.variables():
        data.loc[variable.name[1:],'numShares'] = variable.varValue

    return data

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
   
    for date in stocks[selectedStocks[0]].index: # Ierating through each day for the simulation

        for stock in selectedStocks: # Iterating through each stock to make sure we sell off any stock that we want to

            # Lets sell some stock!
            if stocks[stock].loc[date,:]['predictedAction'] == 'sell':
                if np.any((stockTrades['stock'] == stock) & (stockTrades['sellPrice'].isnull())): # Testing if we have any stocks to sell for this company, if yes continue
                    # Getting the indexes of stocks we need to sell (i.e. rows that contain a null value since they have no)
                    indices = stockTrades[(stockTrades['stock'] == stock) & (stockTrades['sellPrice'].isnull())].index.values
                    stockTrades.loc[indices,'sellPrice'] = stocks[stock].loc[date,:]['Open']

                    # Recording the profit made on this trade
                    moneyOut = np.round(stockTrades.loc[indices, 'numShares'].values[0] * stockTrades.loc[indices, 'buyPrice'].values[0], decimals=2)
                    moneyIn = np.round(stockTrades.loc[indices, 'numShares'].values[0] * stockTrades.loc[indices, 'sellPrice'].values[0], decimals=2)
                    stockTrades.loc[indices,'profit'] = stocks[stock].loc[date,:]['Open'] = moneyIn - moneyOut
                    stockTrades.loc[indices,'sellDate'] = date

                    liquidity = liquidity + moneyIn         

        stockOptions = pd.DataFrame(columns=['stock', 'price','expectedProfit', 'numShares'])
        for stock in selectedStocks: # Iterating through each stock to buy any stock that we want to\
            # Lets buy some stock!
            if stocks[stock].loc[date,:]['predictedAction'] == 'buy': # Testing if we even want to buy this stock today
                price = stocks[stock].loc[date,:]['Open']
                expectedProfit = stocks[stock].loc[date,:]['predictedTomorrow'] - stocks[stock].loc[date,:]['Open']
                stockOptions = stockOptions.append({'stock':stock, 'price':price, 'expectedProfit': expectedProfit}, ignore_index=True)
        
        # Sending the stocks we want to buy to the optimizer to decide which ones we should buy (and how many shows of each)
        stockOptions = optimizer(stockOptions, liquidity, risk)

        # Filtering out the stocks the optimizer said not to buy
        stockOptions = stockOptions[stockOptions['numShares'] > 0]

        for stock, potentialStockPurchase in stockOptions.iterrows():
            numShares = potentialStockPurchase['numShares']
            price = potentialStockPurchase['price']
            liquidity = liquidity - np.round(numShares * price,decimals=2)
            stockTrades = stockTrades.append({'date':date, 'stock':stock, 'buyPrice': price, 'numShares': numShares}, ignore_index=True)


        
    # Getting the cumulative profit by stock
    # CITATION: https://datascience.stackexchange.com/questions/41834/how-to-calculate-cumulative-sum-with-groupby-in-python/41837
    groupCumSum = stockTrades.groupby(['stock', 'date']).sum().groupby('stock').cumsum()
    groupCumSum = groupCumSum.rename(columns={'profit':'groupCumProfit'})
    groupCumSum = groupCumSum[['groupCumProfit']]

    groupProfits = pd.merge(stockTrades, groupCumSum, on=['date', 'stock'])

    groupProfits = groupProfits.set_index('date', drop=True)

    totalProfits = stockTrades.groupby('date').sum('profit')['profit'].cumsum()
    totalProfits = pd.DataFrame(data=totalProfits, columns=['profit'])

    return groupProfits, totalProfits

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

def profitPlot(groupData, totalData, startDate, endDate, stocks):
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)

    # CITATION: https://community.plotly.com/t/plotly-colours-list/11730/2
    DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

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
        if stock == 'all':
            continue
        print(groupData)
        filterData = groupData[(groupData['stock'] == stock) & (groupData.index >= startDate) & (groupData.index <= endDate)]

        print(filterData)

        fig.add_trace(
            go.Scatter(x=filterData.index, 
                y=filterData['groupCumProfit'], 
                name=f"{stock}",
                mode='lines',
                line = dict(color=DEFAULT_PLOTLY_COLORS[i])))
        i = i + 1
    return fig
# Reading in Data
# data = pd.read_csv('Data/dataComplete.csv', index_col = 'Date')
# print(trader('2018-11-19 00:00:00', '2020-11-16 00:00:00', 1000, data, ['ford', 'tesla'], 0.5))

# data = pd.read_csv('Data/dataComplete.csv', index_col = 'Date')

# groupProfits, totalProfits = trader('2018-11-19 00:00:00', '2020-11-16 00:00:00', 1000, data, ['ford', 'tesla'], 0.5)
# groupProfits = groupProfits.set_index('date')
# # totalProfits = totalProfits.set_index('date')

# groupProfits = pd.read_csv('group.csv', index_col='date')
# totalProfits = pd.read_csv('total.csv', index_col='date')
# profitPlot(groupProfits,totalProfits, '2018-11-19 00:00:00', '2020-11-16 00:00:00', ['all','ford', 'tesla'])




