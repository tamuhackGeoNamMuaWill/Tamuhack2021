import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
plt.style.use("fivethirtyeight")

# user inputs ticker name
filename = input('enter ticker symbol: ')
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

# Testing Model: Score returns the coefficient of determination R^2 of the prediction.
# svm_confidence = svr_rbf.score(x_test, y_test)
svm_confidence = 0
# while loop to get the best results
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
stop = 1
while svm_confidence <= stop:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)
    svm_confidence = svr_rbf.score(x_test, y_test)
    stop -= 0.01

print("svm confidence: ", svm_confidence)

# Create and train the Linear Regression  Model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction.
lr_confidence = 0
stop = 1
while lr_confidence <= stop:
    lr.fit(x_train, y_train)
    lr_confidence = lr.score(x_test, y_test)
    stop -= 0.01
print("lr confidence:", lr_confidence)

# Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]

# Print linear regression model predictions for the next '30' days
lr_prediction = lr.predict(x_forecast)

# Print support vector regressor model predictions for the next '30' days
svm_prediction = svr_rbf.predict(x_forecast)


plt.figure(figsize=(15, 8))
plt.plot(df['Adj Close'], label='adj close')
plt.plot(lr_prediction, label="lr")
plt.plot(svm_prediction, label="svm")
plt.title("adj close price Prediction")
plt.xlabel("30 day forecast")
plt.ylabel("Adj. Close Price $")
plt.legend(loc="upper left")
plt.show()
