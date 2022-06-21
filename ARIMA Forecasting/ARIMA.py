import pandas as pd
import numpy as np
from scalecast.Forecaster import Forecaster
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set(rc={'figure.figsize':(14,7)})

df = pd.read_csv('datasets/coin_Bitcoin.csv')
df.drop(['SNo', 'High', 'Low', 'Open', 'Volume', 'Marketcap', 'Name', 'Symbol'],axis=1,inplace=True)

f = Forecaster(y=df['Close'][1400:2700],current_dates=df['Date'][1400:2700])
f.generate_future_dates(12) # 12-month forecast horizon
f.set_test_length(.2) # 20% test set
f.set_estimator('arima') # set arima
f.manual_forecast(call_me='arima1') # forecast with arima

# EDA
f.plot_acf()
plt.show()
f.plot_pacf()
plt.show()
f.seasonal_decompose().plot()
plt.show()
stat, pval, _, _, _, _ = f.adf_test(full_res=True)
print(stat)
print(pval)

warnings.filterwarnings('ignore')

def arima3():
	data = df.set_index('Date')
	train = data.iloc[:int(.8*(df.shape[0])),:]
	auto_model = auto_arima(
	  train,
	  start_P=1,
	  start_q=1,
	  max_p=6,
	  max_q=6,m=12,
	  seasonal=True,
	  max_P=2, 
	  max_D=2,
	  max_Q=2,
	  max_d=2,
	  trace=True,
	  error_action='ignore',
	  suppress_warnings=True,
	  stepwise=True,
	  information_criterion="aic",
	  alpha=0.05,
	  scoring='mse'
	)

	best_params = auto_model.get_params()
	order = best_params['order']
	seasonal_order = best_params['seasonal_order']
	trend = best_params['trend']

	f.manual_forecast(order=order,seasonal_order=seasonal_order,trend=trend,call_me='arima3')

	f.plot_test_set(ci=True,models='arima3')
	plt.title('ARIMA Test-Set Performance',size=14)
	plt.show()

	f.plot(ci=True,models='arima3')
	plt.title('ARIMA Forecast Performance',size=14)
	plt.show()

def arima4():
	f.set_validation_length(12)
	grid = {
		'order':[(1,1,1),(1,1,0),(0,1,1)],
		'seasonal_order':[(2,1,1,12),(1,1,1,12),(2,1,0,12),(0,1,0,12)]
	}

	f.ingest_grid(grid)
	f.tune()
	f.auto_forecast(call_me='arima4')

	f.plot_test_set(ci=True,models='arima4')
	plt.title('ARIMA Test-Set Performance',size=14)
	plt.show()

	f.plot(ci=True,models='arima4')
	plt.title('ARIMA Forecast Performance',size=14)
	plt.show()

	f.regr.summary()

arima4()

print(f.regr.summary())