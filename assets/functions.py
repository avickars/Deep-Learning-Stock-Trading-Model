import pandas as pd
from datetime import date, timedelta
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

def optimizer(startDate, endDate, seedMoney, ford):
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)


    ford = pd.read_csv('Data/fordComplete.csv', index_col='Date')
    ford.index = pd.to_datetime(ford.index)
    ford = ford[ford.index >= startDate]
    ford = ford[ford.index <= endDate]

    stocks = {'ford':ford}

    # Initialize df to hold all the stocks bought
    stockTrades = pd.DataFrame(columns = ['date', 'stock', 'buyPrice', 'numShares', 'sellPrice', 'profit'])
    
    # Initialize variable to hold the cash we currently have to spend
    liquidity = seedMoney

    i = 1
    for date in ford.index: # Ierating through each day
        for stock in stocks.keys(): # Iterating through each stock

            # Lets sell some stock!
            if stocks[stock].loc[date,:]['action'] == 'sell':
                print("Lets sell!")
                print(liquidity)
                print(date)
                if len(stockTrades[stockTrades.isnull().any(axis=1)]) > 0: # Testing if we own any stocks of this company, if yes continue and sell them
                    # Recording the price the stock is sold at
                    stockTrades.loc[stockTrades[stockTrades.isnull().any(axis=1)].index.values,'sellPrice'] = stocks[stock].loc[date,:]['Open']

           
                    # Getting the indexes of stocks we need to sell (i.e. rows that contain a null value since they have no)
                    indices = stockTrades[stockTrades.isnull().any(axis=1)].index.values

                    # Recording the profit made on this trade
                    moneyOut = stockTrades.loc[indices, 'numShares'].values[0] * stockTrades.loc[indices, 'buyPrice'].values[0]
                    moneyIn = stockTrades.loc[indices, 'numShares'].values[0] * stockTrades.loc[indices, 'sellPrice'].values[0]
                    stockTrades.loc[indices,'profit'] = stocks[stock].loc[date,:]['Open'] = moneyIn - moneyOut
                    
                    # Updating the liquidity value to reflect the stocks we just sold
                    liquidity = liquidity + moneyIn

            # Lets buy some stock!
            if stocks[stock].loc[date,:]['action'] == 'buy':
                price = stocks[stock].loc[date,:]['Open']
                numShares = np.floor(liquidity/price)
                if numShares > 0: # Testing if we have enough liquidity to buy any shares, if yes continue otherwise skip
                    liquidity = liquidity - numShares * price
                    stockTrades = stockTrades.append({'date':date, 'stock':stock, 'buyPrice': price, 'numShares': numShares}, ignore_index=True)
        # i  = i + 1
        # if i > 400:
        #     break
    print(stockTrades)
    print(stockTrades['profit'].sum())


optimizer('2018-11-19 00:00:00', '2020-11-16 00:00:00', 1000, 'dolthis')



