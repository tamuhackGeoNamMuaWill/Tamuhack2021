# importing cool packages
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd

ticker_list=['DJIA', 'DOW', 'TSLA', 'LB', 'EXPE', 'PXD', 'MCHP', 'CRM', 'JEC', 'NRG', 'HFC', 'NOW']

today = date.today()

start_date = "2021-01-01"
end_date = "2021-01-21"

files = []
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

for i in range(0,11): #reading csv files
    df1 = pd.read_csv('./data/' + str(files[i]) + '.csv')
    print(df1.head())

#Take User input
# Import Data
# Manipulating CSV File
# Calculate Average for 20 days and 5 days
# Plot Moving Average
# Support and Resistence
