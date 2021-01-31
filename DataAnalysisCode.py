# importing cool packages
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd

files = []
files_SMA = []
ticker_list = []
#Dowwnloading Files
def filesDownload():
    global ticker_list
    global files
    global files_SMA
    today = date.today()

    start_date = "2016-01-01"
    end_date = today

    def getData(ticker): #downloading data
        print(ticker)
        data=pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
        dataname = ticker + '_' + str(start_date) +'-'+ str(end_date)
        files.append(dataname)
        SaveData(data, dataname)

    def SaveData(df, filename):
        df.to_csv('./data/' + filename + ".csv")

    def SMA(filename):
        df = pd.read_csv("data/" + filename + ".csv")
        df['SMA_200'] = df.iloc[:, 5].rolling(window=200).mean()
        df['SMA_50'] = df.iloc[:, 5].rolling(window=50).mean()
        dataname = filename + "_with_SMA"
        files_SMA.append(dataname)
        SaveData(df, dataname)

    for tik in ticker_list:
        getData(tik)
    for i in files:
        SMA(i)

#Take User input
def userInput():
    ticker = input("Please enter a list of ticker you want to keep track of separated by spaces: \n")
    global ticker_list
    ticker_list = ticker.split()


######### RUNING CODE ########
userInput()
filesDownload()
