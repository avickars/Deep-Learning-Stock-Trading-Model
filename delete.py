import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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

    filterData = data[(data['stock'] == 'ford') & (data.index >= startDate) & (data.index <= endDate)]

    iris = px.data.iris()
    fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
    df = pd.DataFrame({
        'x':[1,2,3,4],
        'y':[5,6,7,8],})
    fig2 = px.bar(df, x="x", y="y")
    fig.add_trace(fig2.data[0])
    fig.show()

    # fig = go.Figure()
    i = 0
    # for stock in stocks:
    #     filterData = data[(data['stock'] == stock) & (data.index >= startDate) & (data.index <= endDate)]

    #     fig.add_trace(
    #         go.Scatter(x=filterData.index, 
    #             y=filterData['Open'], 
    #             name=f"{stock} - Truth",
    #             mode='lines',
    #             line = dict(color=DEFAULT_PLOTLY_COLORS[i])))

    #     fig.add_trace(
    #         go.Scatter(x=filterData.index, 
    #             y=filterData['predicted'],
    #             name=f"{stock} - Predicted",
    #             line = dict(dash='dot', width=3, color=DEFAULT_PLOTLY_COLORS[i])))
    #     i = i + 1
    fig.show()


ford = pd.read_csv('Data/fordComplete.csv', index_col = 'Date')

# trader('2018-11-19 00:00:00', '2020-11-16 00:00:00', 1000, 'dolthis')
predictionPlot(ford, '2020-10-19 00:00:00', '2020-11-16 00:00:00',["ford"])




