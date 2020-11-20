import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import  model_from_json
import tensorflow as tf

def transform(df, n=40):
    df = pd.DataFrame(df['Open'])  # Removing all the other columns as we are only predicting if we should buy based on the opening stock price

    
    # Normalizing to 0-1 range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=pd.to_datetime(df.index))


    # Creating 40 columns that give the past 40 day opening stock price
    for i in range(1, n + 1):
        df[f'Open-{i}'] = df['Open'].shift(i)

    # Subsetting for the neccessary columns
    df = df.iloc[40:, 0:]

    # Splitting into training and testing data (test size is about last 2 years)
    dt = pd.to_datetime(date(2020, 11, 17) - timedelta(days=730))
    train = df[df.index < dt]
    test = df[df.index >= dt]


    # Splitting into appropriat x and y values
    xTrain = train.iloc[:, 1:]
    yTrain = train.iloc[:, 0]
    xTest = test.iloc[:, 1:]
    yTest = test.iloc[:, 0]

    # Converting to numpy arrays to feed into model
    xTrain = xTrain.to_numpy()
    yTrain = yTrain.to_numpy()
    xTest = xTest.to_numpy()
    yTest = yTest.to_numpy()

    # Reshaping to get correct form
    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

    return xTrain, yTrain, xTest, yTest, scaler

def loadModel(stockName):
    # CITATION: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # load json and create model
    json_file = open(f'data/{stockName}EpochTraining2000.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(f"data/{stockName}EpochTraining2000.h5")
    return model


# Calculates the predicted vales and residuals
def prediction(scaler, model, x, y):
    with tf.device("cpu:0"): # ******************** DELETE THIS, WAS HERE BECAUSE GPU WAS BUSY
        preds = model.predict(x)
        preds = scaler.inverse_transform(preds)
        unscaled_yTest = scaler.inverse_transform(np.reshape(y, (-1, 1)))
        return (preds-unscaled_yTest), preds, unscaled_yTest

def formulateData(originalData, residuals, predicted, saveAs):
    originalData = pd.DataFrame(originalData['Open'])

    originalData['residuals'] = residuals
    originalData['predicted'] = predicted

    originalData['tomorrow'] = originalData['predicted'].shift(-1)

    originalData['action'] = np.where(originalData['tomorrow'] - originalData['Open'] > 0, 'buy', 'sell')

    originalData.to_csv(saveAs)


def main():
    # FORD
    ford = pd.read_csv("Data/ford.csv", index_col='Date')
    ford = pd.DataFrame(ford, index=pd.to_datetime(ford.index))

    model = loadModel("ford")
    xTrain, yTrain, xTest, yTest, scaler = transform(ford, 40)
    
    dt = pd.to_datetime(date(2020, 11, 17) - timedelta(days=730))
    fordTest = ford[ford.index >= dt]

    residuals, yHat, y = prediction(scaler, model, xTest, yTest)

    formulateData(fordTest, residuals, yHat, 'Data/fordComplete.csv')






if __name__ == '__main__':
    main()