import pandas as pd
from numpy import mean
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv('datasets/coin_Bitcoin.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.drop(['SNo', 'High', 'Low', 'Open', 'Volume', 'Marketcap', 'Name', 'Symbol'],axis=1,inplace=True)

mas = 1

smoothing = 10

# prepare situation
rolling = df.rolling(window=smoothing)
rolling_mean = rolling.mean()

if mas:
	X = rolling_mean.values[smoothing:]#df.values
else:
	X = df.values

window = 10
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
for t in range(len(test)):
	length = len(history)
	yhat = mean([history[i] for i in range(length-window,length)])
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test[1000:])
pyplot.plot(predictions[1000:], color='red')
pyplot.show()
# zoom plot
pyplot.plot(test[2500:2950])
pyplot.plot(predictions[2500:2950], color='red')
pyplot.show()