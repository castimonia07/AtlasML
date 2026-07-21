import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

# Generate synthetic time series
np.random.seed(42)
y = pd.Series(np.random.randn(200).cumsum())
exog = pd.DataFrame(np.random.randn(200, 10), columns=[f"col_{i}" for i in range(10)])

print("Testing ARIMA default fit:")
t0 = time.time()
model = ARIMA(y, exog=exog, order=(2, 1, 2))
res = model.fit()
print(f"Default fit took: {time.time() - t0:.4f}s")

print("Testing ARIMA optimized fit (maxiter=20, cov_type='none'):")
t0 = time.time()
model = ARIMA(y, exog=exog, order=(2, 1, 2))
res = model.fit(method_kwargs={'maxiter': 20}, cov_type='none')
print(f"Optimized fit took: {time.time() - t0:.4f}s")

print("Testing SARIMAX default fit:")
t0 = time.time()
model = SARIMAX(y, exog=exog, order=(2, 1, 2), seasonal_order=(0, 1, 0, 7))
res = model.fit(disp=False)
print(f"Default fit took: {time.time() - t0:.4f}s")

print("Testing SARIMAX optimized fit (maxiter=20, cov_type='none'):")
t0 = time.time()
model = SARIMAX(y, exog=exog, order=(2, 1, 2), seasonal_order=(0, 1, 0, 7))
res = model.fit(maxiter=20, cov_type='none', disp=False)
print(f"Optimized fit took: {time.time() - t0:.4f}s")
