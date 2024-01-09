---
title: AI/ML Project - Stock Price Prediction Using LSTM Neural Networks
date: 2023-11-10
tags: 
image : "/coverimg/Neural_Network.png"
Description  : This program uses an artificial recurrent neural network...
draft: false
---

**Project Overview**
This project showcases my skills in machine learning and data analysis by predicting the closing stock price of Apple Inc. using an LSTM neural network. The project demonstrates my ability to work with financial data and advanced machine learning techniques.

**Key Features**

- **Data Source:** Utilizes historical stock data from Yahoo Finance, accessed via the **`yfinance`** library.
- **Data Preprocessing:** Implements data cleaning and normalization to prepare the dataset for the LSTM model.
- **Model Architecture:** Employs an LSTM neural network, known for its effectiveness in handling time-series data.
- **Visualization:** Includes data visualization for stock price trends and model performance using **`matplotlib`**.

**Technical Stack**

- **Python Libraries:** **`numpy`**, **`pandas`**, **`yfinance`**, **`keras`**, **`sklearn`**, **`matplotlib`**
- **Machine Learning:** Long Short Term Memory (LSTM) neural network for time-series prediction.
- **Data Handling:** Pandas DataFrames for data manipulation and preprocessing.

**Challenges and Learnings**

- **Data Preprocessing:** Addressed challenges in cleaning and normalizing financial time-series data.
- **Model Tuning:** Explored various hyperparameters to optimize the LSTM model for stock price prediction.
- **Visual Analysis:** Gained insights into the importance of visualizing data trends in financial analysis.

**Project Outcome**
Successfully developed a model that predicts the closing price of Apple Inc. stock with a reasonable degree of accuracy. This project highlights my ability to apply machine learning techniques to real-world financial data.


### Let me walkthru my steps
Description: This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM) to predict the closing stock price of a corporation (Apple Inc.) using the past 60 day stock price. 

- Let's import all the neccessaries libraires
```
#Import the libraries
#import pandas_datareader as web
import math
import numpy as np
import pandas_datareader as pdr
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as p1t
p1t.style.use('fivethirtyeight')
```

- Get the stock quote, here I am using yahoo finance historical data
```
#Get the stock quote
df = yf.download('AAPL', start='2016-01-01', end='2023-12-13')
#Show the data
df
```
Run this code and you should get the list of data.
```
[*********************100%%**********************]  1 of 1 completed
Open	High	Low	Close	Adj Close	Volume
Date						
2016-01-04	25.652500	26.342501	25.500000	26.337500	23.977478	270597600
2016-01-05	26.437500	26.462500	25.602501	25.677500	23.376617	223164000
2016-01-06	25.139999	25.592501	24.967501	25.174999	22.919140	273829600
2016-01-07	24.670000	25.032499	24.107500	24.112499	21.951855	324377600
2016-01-08	24.637501	24.777500	24.190001	24.240000	22.067926	283192000
...	...	...	...	...	...	...
2023-12-06	194.449997	194.759995	192.110001	192.320007	192.320007	41089700
2023-12-07	193.630005	195.000000	193.589996	194.270004	194.270004	47477700
2023-12-08	194.199997	195.990005	193.669998	195.710007	195.710007	53377300
2023-12-11	193.110001	193.490005	191.419998	193.179993	193.179993	60943700
2023-12-12	193.080002	194.720001	191.720001	194.710007	194.710007	52696900
2000 rows × 6 columns
```
***note***
We query data from January 1st, but the data shows January 4th as start. The market opens on January 4th of that year. The data also includes "open", "high", "low", "Cloase"...etc. We will only want show the "Close" price in the following steps.

- Let's find out how many number of rows and cloumns in this data set first.
```
#Get the number of rows and columns in the data set
df.shape
```
Result: (2002,6)

- I want to visualize the closing prices history in the chart first to see how it looks like
```
#Visualize the closing prices history
p1t.figure(figsize=(16,8))
p1t.plot(df['Close'])
p1t.title('APPL Close Price History')
p1t.xlabel('Date', fontsize=18), p1t.ylabel('Close Price USD($)', fontsize=18)
p1t.show()
```
Result:

![APPL Stock Closing Price!](/img/stock_0.jpg "APPL Closing Price History")

- Next, create a new dataframe with only the "Close column", convert dataframe to numpy array.
- I also want to get the number of rows to train the model on.
```
#Create a new dataframe with only the 'Close column'
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil( len(dataset) * .8)

training_data_len
```
Result: 1602

- 