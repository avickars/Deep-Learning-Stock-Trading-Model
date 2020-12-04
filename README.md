# CMPT 353 Stock Market Predictor

This README document details what each folder, and file contains as well as what they do and how to run them.

## Scripts

#### neuralNetwork.ipynb

##### Description

This file contains the code that tunes the parameters of the chosen model on the stock data for Forward Industries.  The code that actually fits a model has been commented out and replaced with a code that reads in the model.  This is because actually fitting a model takes a significant amount of time.

##### How to run it
```python
python neuralNetwork.ipynb
```
##### Required Packages

* datetime
* numpy
* pandas
* keras
* matplotlib
* tensorflow

#### neuralNetworkAnalysis.ipynb

##### Description

This file contains the code that analyses the results of the model when applied on Forward Industries, and the subsequent boosting.

##### How to run it
```python
python neuralNetworkAnalysis.ipynb
```
##### Required Packages

* datetime
* numpy
* pandas
* keras
* matplotlib
* tensorflow

#### {companyName}Model.ipynb

##### Description

Each of these Jupyter notebooks contains the fitting of the LSTM Neural Network for each respective company.  The code that actually fits a model has been commented out and replaced with a code that reads in the model.  This is because actually fitting a model takes a significant amount of time.

##### How to run it
```python
python {companyName}Model.ipynb
```

##### Required Packages

* datetime
* numpy
* pandas
* keras
* matplotlib
* tensorflow

#### app.py

##### Description

This python file is the dashboard that has been created to vizualize the results of the model.  To run this, run the python file using "python app.py", and navigate to the server location that is running locally that will appear in the command line.  For additional clarification, please see the image linked below.

#### How to run it

1. Enter the Command Line Command
```python
python app.py
```

2. Click the indicated Link
![Intial Startup Image](https://drive.google.com/file/d/1GXyUv3Yyr1oNyWF3LiUVMf6oHuciexmt/view?usp=sharing)

3. Wait (approximately 20-30seconds depending on computer speed).  Ignore the indicated error (it is to do with the optimization model, I have not been able to disable it)

![Ignore This Error](https://drive.google.com/file/d/1GXyUv3Yyr1oNyWF3LiUVMf6oHuciexmt/view?usp=sharing)

4. Adjust Simulation Paraters as Desired

Initial Parameters Are:
* Stocks: All Companies
* Time Range: November 19, 2018 - November 16, 2020 (Max Time Range)
* Seed Money: $1000
* Risk: 1

To change Parameters See the Picture Linked Below

##### Required Packages

* dash
* dash-bootstrap-components
* plotly
* pulp
* numpy
* pandas