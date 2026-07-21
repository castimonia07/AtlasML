import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.simplefilter('ignore')

np.random.seed(42)
y = pd.Series(np.random.randn(100).cumsum())
exog = pd.DataFrame(np.random.randn(100, 5), columns=[f"col_{i}" for i in range(5)])

model = ARIMA(y, exog=exog, order=(1, 1, 1))
res = model.fit()

# Test predict using predict(exog) or similar
print("Testing predict with exog:")
try:
    preds = res.predict(exog=exog.iloc[:10])
    print("res.predict(exog=...) worked! Length:", len(preds))
except Exception as e:
    print("res.predict(exog=...) failed:", e)

try:
    # First argument in predict is usually `start`. Let's see if passing a DataFrame raises error
    preds = res.predict(exog.iloc[:10])
    print("res.predict(df) worked! Length:", len(preds))
except Exception as e:
    print("res.predict(df) failed:", e)
