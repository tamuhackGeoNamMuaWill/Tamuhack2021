import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import seaborn as sns
from datetime import date
from pandas_datareader import data as pdr
sns.set_style("whitegrid")

files = []
files_SMA = []
ticker_list = []


def filesDownload():
    global ticker_list
    global files
    global files_SMA
    today = date.today()

    start_date = "2016-01-01"
    end_date = today

    def getData(ticker):  # downloading data
        data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
        # dataname = ticker + '_' + str(start_date) + '-' + str(end_date)
        data['SMA_200'] = data.iloc[:, 5].rolling(window=200).mean()
        data['SMA_50'] = data.iloc[:, 5].rolling(window=50).mean()
        files_SMA.append(ticker)
        files.append(ticker)
        SaveData(data, ticker)

    def SaveData(df, filename):
        # df.to_csv('./data/' + filename + ".csv")
        dnew = df.iloc[200:]
        dnew.to_csv('_ticker' + '.csv')

    # def SMA(filename):
    #     df = pd.read_csv(filename + ".csv")
    #     df['SMA_200'] = df.iloc[:, 5].rolling(window=200).mean()
    #     df['SMA_50'] = df.iloc[:, 5].rolling(window=50).mean()
    #     dataname = filename + "_with_SMA"
    #     files_SMA.append(filename)
    #     SaveData(df, filename)

    #for tik in ticker_list:
        #getData(tik)
    getData(ticker_list)
    # for i in files:
    #     SMA(i)


def userInput():
    ticker = input("Please enter a ticker symbol: ").upper()
    global ticker_list
    # ticker_list = ticker.split()
    ticker_list = ticker


def prediction(filename):
    # filename = input('enter ticker symbol: ')
    df = pd.read_csv(filename + '.csv')

    # Remove the date
    del df['Date']

    # A variable for predicting 'n' days out into the future
    forecast_out = 30  # 'n=30' days
    # Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    # print(df.tail())

    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'], 1))

    # Remove the last '30' rows
    X = X[:-forecast_out]

    # Convert the data frame to a numpy array
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]

    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and train the Support Vector Machine (Regressor)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)
    svr_confidence = 0
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    stop = 1
    # while loop to get the best results
    while svr_confidence <= stop:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr_rbf.fit(x_train, y_train)
        svr_confidence = svr_rbf.score(x_test, y_test)
        stop -= 0.01

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_confidence = 0
    stop = 1
    # while loop to get the best results
    while lr_confidence <= stop:
        lr.fit(x_train, y_train)
        lr_confidence = lr.score(x_test, y_test)
        stop -= 0.01

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]

    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)
    old = df[['Adj Close']]
    for x in lr_prediction:
        new_row = {'Open': 0, 'High': 0, 'Low': 0, 'Close': 0, 'Adj Close': x, 'Volume': 0}
        df = df.append(new_row, ignore_index=True)

    # svm_prediction = svr_rbf.predict(x_forecast)

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(15, 11))
    plt.xlim([len(df) - 100, len(df)-1])
    plt.ylim([(df['Adj Close'].tail(100).min() - df['Adj Close'].tail(100).min() * 0.1), (df['Adj Close'].tail(100).max() + df['Adj Close'].tail(100).max() * 0.1)])
    plt.plot(df['Adj Close'], color='red', label='Predicted Price')
    plt.plot(old['Adj Close'], color='k', label="Past Data")
    plt.plot(df['SMA_200'], color='b', label='SMA 200')
    plt.plot(df['SMA_50'], color='g', label='SMA 50')
    # plt.yticks(np.arange(int(df['Adj Close'].tail(100).min() * 1.1), int(df['Adj Close'].tail(100).max() * 1.1), step=(int(df['Adj Close'].tail(100).max() * 1.1))/10))
    plt.title("30 Day Prediction of " + filename)
    plt.xlabel("Days")
    plt.ylabel("Adj. Close Price $")
    plt.legend()
    plt.savefig("_graph")
    # plt.show()


userInput()
filesDownload()
prediction(ticker_list)
