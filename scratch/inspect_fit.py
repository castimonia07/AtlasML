from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import inspect

print("ARIMA.fit parameters:")
print(inspect.signature(ARIMA.fit))

print("\nSARIMAX.fit parameters:")
print(inspect.signature(SARIMAX.fit))
