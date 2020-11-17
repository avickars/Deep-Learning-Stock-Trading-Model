# This script downloads the stock data for various var companies via the yfinance package that downloads the data from Yahoo Finance
# The github page for this package is here: https://github.com/ranaroussi/yfinance

# INSTRUCTIONS: How this works is you enter the short-hand name of the company, and search on that.  It will return the open/close prices and some other stuff
# for the last 10 years. To find the shorthand name of a company, search it on yahoo finances (https://ca.finance.yahoo.com/)
# and you'll get the shorthand name.  There is other information you can get. see the github page.   Essentially all the information on the yahoo
# finance page you can get via this package.


import yfinance as yf


def getData(company):
    object = yf.Ticker(company)
    return object.history(start="2000-01-01", end="2020-11-17")


def main():
    getData('FORD').to_csv('Data/ford.csv')

    getData('FCAU').to_csv('Data/fiatChrysler.csv')

    getData('TSLA').to_csv('Data/tesla.csv')

    getData('GM').to_csv('Data/generalMotors.csv')

    getData('HMC').to_csv('Data/honda.csv')


if __name__ == '__main__':
    main()
