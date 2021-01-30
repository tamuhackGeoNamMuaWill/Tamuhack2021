# importing cool packages
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd

ticker_list = []
files = []

#Dowwnloading Files
def filesDownload(stocks):
    today = date.today()

    start_date = "2016-01-01"
    end_date = date.today()

    global files
    def getData(ticker): #downloading data
        print(ticker)
        data=pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
        dataname = ticker + '_' + str(start_date) +'-'+ str(end_date)
        files.append(dataname)
        SaveData(data, dataname)

    def SaveData(df, filename):
        df.to_csv('./data/' + filename + ".csv")

    for tik in ticker_list:
        getData(tik)

    # for i in range(0,12): #reading csv files
    #     df1 = pd.read_csv('./data/' + str(files[i]) + '.csv')
    #     print(df1.head())

#Take User input
def userInput():
    ticker = input("Please enter a list of ticker you want to keep track of separated by spaces: \n")
    global ticker_list
    ticker_list = ticker.split()

# Import Data

# Calculate Average for 20 days and 5 days
def SMA(filename):
    df = pd.read_csv("data/" + filename + ".csv")
    df['SMA_200'] = df.iloc[:,5].rolling(window=len(df) - 200).mean()
    df['SMA_100'] = df.iloc[:, 5].rolling(window=len(df) - 100).mean()
    df['SMA_50'] = df.iloc[:, 5].rolling(window=len(df) - 50).mean()
    df['SMA_20'] = df.iloc[:, 5].rolling(window=len(df) - 20).mean()
    df['SMA_10'] = df.iloc[:, 5].rolling(window=len(df) - 10).mean()
    df['SMA_5'] = df.iloc[:, 5].rolling(window=len(df) - 5).mean()
    df['SMA_1'] = df.iloc[:, 5].rolling(window=len(df) - 1).mean()
    df.to_csv("data/" + filename +"_with_SMA"+".csv")

# Plot Moving Average

# Support and Resistence


######### RUNING CODE ########
ticker_list = ['AAPL','TSLA','GOOG']

filesDownload(ticker_list)
for i in files:
    SMA(i)

