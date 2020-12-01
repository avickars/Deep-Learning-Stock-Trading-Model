from datetime import date, timedelta
import numpy as np
import pandas as pd
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# Delete this ***********************************
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# Calculates the MSE 
def mse(x, y, model, scaler):
    preds = model.predict(x)
    preds = scaler.inverse_transform(preds)
    unscaled_yTest = scaler.inverse_transform(np.reshape(y, (-1, 1)))
    return np.mean(np.square(preds-unscaled_yTest)), preds, unscaled_yTest

# This function accepts the raw data and transforms it into the form required for the LSTM NN.  
# - It removes unneccessary columns
# - scales the data between 0 and 1
# - transforms into numpy arrays
def transform(df, n=40):
    df = pd.DataFrame(df['Open'])  # Removing all the other columns as we are only predicting if we should buy based on the opening stock price

    
    # Normalizing to 0-1 range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=pd.to_datetime(df.index))


    # Creating 40 columns that give the past 40 day opening stock price
    for i in range(1, n + 1):
        df[f'Open-{i}'] = df['Open'].shift(i)

    # Subsetting for the neccessary columns.  Chopping off the 1st 40 days as they contain null values (i.e. the data set doesn't contain the past 40 days of opening stock prices for them)
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
    
# This function saves a model
def saveModel(model, name, location='Data'):
    # CITATION: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # serialize model to JSON
    model_json = model.to_json()
    with open(f"{location}/{name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{location}/{name}.h5")
    print("Saved model to disk")

# This function loads a model
def loadModel(name, location='Data'):
    # CITATION: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # load json and create model
    json_file = open(f"{location}/{name}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(f"{location}/{name}.h5")
    return model
    print("Loaded model from disk")

# This function trains a model
def modelTraining(epoch, batchSize, xTrain,yTrain, neuronsLSTM1=50,neuronsDense=64, learningRate=0.0005):
    model = Sequential()
    model.add(LSTM(neuronsLSTM1,input_shape=(xTrain.shape[1],xTrain.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(neuronsDense))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    opt = tf.keras.optimizers.Adam(lr=learningRate)
    model.compile(optimizer=opt, loss='mse')
    model.fit(xTrain,yTrain, batch_size=batchSize, epochs=epoch, shuffle=True)
    return model

# This function takes the originalData (train or test), the residuals, and predicted and formualtes it into a df that contains our predicted action
def formulateData(originalData, residuals, predicted):
    originalData = pd.DataFrame(originalData['Open'])

    originalData['residuals'] = residuals
    originalData['predicted'] = predicted

    originalData['tomorrow'] = originalData['predicted'].shift(-1)

    originalData['action'] = np.where(originalData['tomorrow'] - originalData['Open'] > 0, 'buy', 'sell')

    return originalData

# This function takes the scaler, mode and x and y values and creates the predicted results.  (Similar to the MSE function above except it doesn't return the MSE)
def prediction(scaler, model, x, y):
    preds = model.predict(x)
    preds = scaler.inverse_transform(preds)
    unscaled_yTest = scaler.inverse_transform(np.reshape(y, (-1, 1)))
    return (preds-unscaled_yTest), preds, unscaled_yTest

# This function accepts the raw data, model and scaler and produces the predicted results in  a usable dataframe
def createUsableDF(data, scaler, model, LAG=40):
    # Making sure we have the date as a datetime in the index
    data = pd.DataFrame(data, index=pd.to_datetime(data.index))

    # Getting the data into numpy arrays with the required shape
    xTrain, yTrain, xTest, yTest, scaler = transform(data, LAG)
    
    # Creating the date object to splot the data on
    dt = pd.to_datetime(date(2020, 11, 17) - timedelta(days=730))
    
    # Creating Training Data
    dataTrain = data[data.index < dt].iloc[40:,:]
    
    # Creating Test Data
    dataTest = data[data.index >= dt]
    
    # Gets the unscaled residuals, predicted values (yHat), and truth (y) for the training data
    residuals, yHat, y = prediction(scaler, model, xTest, yTest)
    
    # Takes the data from prediction() and creates a df
    tradingActionsTest = formulateData(dataTest, residuals, yHat)

    # Gets the unscaled residuals, predicted values (yHat), and truth (y) for the training data
    residuals, yHat, y = prediction(scaler, model, xTrain, yTrain)
    
    # Takes the data from prediction() and creates a df
    tradingActionsTrain = formulateData(dataTrain, residuals, yHat)

    return tradingActionsTrain, tradingActionsTest

# This function boosts the model accuracy by amending the predicted values by subtracting from it the average residuals from the last 40 days
def boostModel(tradingActionsTrain, tradingActionsTest):
    # Defining how far back we will use the residuals from
    lookBack = 180
    # Concating the 2 dfs into one for ease of use
    tradingActionsTest_full = pd.concat([tradingActionsTrain.iloc[-lookBack:,],tradingActionsTest])
    movingResidualAvg = []
    #iterating for each day in the test set
    for i in range(lookBack, len(tradingActionsTest_full)):
        # Taking the average residual for the last 180 days (exclusive i.e. of course not using todays residual since that would be cheating)
        residualAverage = np.average(tradingActionsTest_full.iloc[i-lookBack:i,:]['residuals'])
        # Adding it to the list
        movingResidualAvg.append(residualAverage )
    # Storing the moving residual
    tradingActionsTest['movingResidual'] = movingResidualAvg
    # Creating the new predicted Value
    tradingActionsTest['predictedTomorrow'] = tradingActionsTest['tomorrow'] - tradingActionsTest['movingResidual']
    # Creating the new action
    tradingActionsTest['action'] = np.where(tradingActionsTest['predictedTomorrow'] - tradingActionsTest['Open']>0,
                                                'buy',
                                                'sell')
    return tradingActionsTest

# This function accepts a df with columns ['Open', 'residuals', 'predicted', 'tomorrow', 'action'], returns the proportion of times we predict correctly that we should
# buy or sell a stock
def score(df):
    df['trueTomorrow'] = df['Open'].shift(-1)
    df['trueAction'] = np.where(df['trueTomorrow']-df['Open']>0,
                                           'buy',
                                           'sell')
    scoreValue = np.sum(df['action'] ==   df['trueAction'])/len(df)
    return scoreValue

# Plots the training and test results of a model. 
def plot(trainTruth, trainPreds, testTruth, testPreds, title):
    # CITATION: https://riptutorial.com/matplotlib/example/11257/grid-of-subplots-using-subplot
    fig, axes = plt.subplots(2, figsize=(8, 6))

    # Set the title for the figure
    fig.suptitle(title, fontsize=15)

    # Top Left Subplot
    axes[0].plot(trainTruth)
    axes[0].plot(trainPreds)
    axes[0].set_title("Training Data")
    axes[0].legend(['Real', 'Predicted'])

    # Top Right Subplot
    axes[1].plot(testTruth)
    axes[1].plot(testPreds)
    axes[1].set_title("Validation Data")
    axes[1].legend(['Real', 'Predicted'])
    fig.show()




































