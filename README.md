# CMPT 353 Stock Market Predictor

This README document details what each folder, and file contains as well as what they do and how to run them.

NOTE: All code has been tested on CSIL.

NOTE: The order of the scripts listed below is ordered according to how it should be run from the start of the model development to the finish. HOWEVER, all scripts are set up so that they can be run independent of order.  For instance, all data they load has all ready been saved in this repository so you don't need to run one script to generate the data for another.

## Scripts

### neuralNetwork.ipynb

#### Description

This file contains the code that tunes the parameters of the chosen model on the stock data for Forward Industries.  The code that actually fits a model has been commented out and replaced with a code that reads in the model.  This is because actually fitting a model takes a significant amount of time.

#### How to run it
```python
python neuralNetwork.ipynb
```
#### Required Packages

* datetime
* numpy
* pandas
* keras
* matplotlib
* tensorflow
* yahoo-finance (Only if running commented out code)
* scikit-learn

### neuralNetworkAnalysis.ipynb

#### Description

This file contains the code that analyses the results of the model when applied on Forward Industries, and the subsequent boosting.

#### How to run it
```python
python neuralNetworkAnalysis.ipynb
```
#### Required Packages

* datetime
* numpy
* pandas
* keras
* matplotlib
* tensorflow
* scikit-learn

### {companyName}Model.ipynb

#### Description

Each of these Jupyter notebooks contains the fitting of the LSTM Neural Network for each respective company.  The code that actually fits a model has been commented out and replaced with a code that reads in the model.  This is because actually fitting a model takes a significant amount of time.

#### How to run it
```python
python {companyName}Model.ipynb
```

#### Required Packages

* datetime
* numpy
* pandas
* keras
* matplotlib
* tensorflow
* yahoo-finance (Only if running commented out code)
* scikit-learn

### formattingData.ipynb

#### Description

The jupyter notebook contains the code the takes the predictions from each model and concatenates them into a single dataframe and outputs it.

#### Required Packages

* Pandas

### app.py

#### Description

This python file is the dashboard that has been created to vizualize the results of the model.  To run this, run the python file using "python app.py", and navigate to the server location that is running locally that will appear in the command line.  For additional clarification, please see the image linked below.

#### How to run it

1. Enter the Command Line Command
```python
python app.py
```

2. Click the indicated Link
![Initial Startup Image](https://github.com/avickars/Deep-Learning-Stock-Trading-Model/blob/main/readMe%20Images/runAppPY.PNG)

3. Wait (approximately 20-30seconds depending on computer speed).  Ignore the indicated error (it is to do with the optimization model, I have not been able to disable it)

![Ignore This Error Image](https://csil-git1.cs.surrey.sfu.ca/avickars/cmpt-353-stock-market-predictor/-/blob/728f5542e3b04b8c6545c5dc1a78d273e773e65d/readMe%20Images/error.PNG)

4. Adjust Simulation Paraters as Desired

Initial Parameters Are:
* Stocks: All Companies
* Time Range: November 19, 2018 - November 16, 2020 (Max Time Range)
* Seed Money: $1000
* Risk: 1

To change Parameters See the Picture Linked Below

![Adjust Parameters Image](https://csil-git1.cs.surrey.sfu.ca/avickars/cmpt-353-stock-market-predictor/-/blob/728f5542e3b04b8c6545c5dc1a78d273e773e65d/readMe%20Images/parameters.PNG)

#### Required Packages

* dash
* dash-bootstrap-components
* plotly
* pulp
* numpy
* pandas

## Folders

### Data

This folder contains all the data, and models used in the scripts above.  The subfolders are:

* "Raw Data" contains the raw data as initially downloaded.
* "Model Training Results" contains the results of each tuning attempt.
* "Final Predictions" contains the data as referenced in app.py
* "Test Models" contains the models created while tuning the Epoch and Batch size
* "CVTraining" contains the models created while tuning the learning rate, and the number of neurons in each layer
* "Final Models" contains the final models for each company

### columns

Contains additional code for app.py (the dashboard).  These were create to modulize the code for app.py

### assets

Contains css code for app.py (the dashboard)

### Project_functions

Contains functions used in the analysis and dashboard

#### analysis_functions

Contains functions used in the jupyter notebooks during the model development.

#### dashboard_functions

Contains functions used in app.py (and leftColumn.py and rightColumn.py).

#### readME Images

Contains the images linked in this read me document.

