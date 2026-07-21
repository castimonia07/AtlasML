import os
import time
import uuid
import datetime as dt

import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.core.config import MODELS_DIR, REPORTS_DIR
from app.core.database import SessionLocal
from app.models.models import Experiment
from app.services.mlflow_utils import init_mlflow


def update_progress(experiment_id: int, progress: str, log_msg: str):
    db = SessionLocal()
    should_raise = False
    try:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            if exp.status in ("stopped", "failed"):
                should_raise = True
            else:
                exp.pipeline_progress = progress
                ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_logs = exp.pipeline_logs or ""
                exp.pipeline_logs = current_logs + f"[{ts}] {progress.upper()}: {log_msg}\n"
                db.commit()
    except Exception as e:
        print(f"Error updating progress: {e}")
    finally:
        db.close()
    if should_raise:
        raise ValueError("Training stopped by user.")


def calculate_mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class ProphetRegression:
    """
    A custom Prophet-like forecasting model using Piecewise Linear Trend
    and Fourier Series for Daily, Weekly and Yearly Seasonality.
    """
    def __init__(self, changepoint_prior_scale=0.05, weekly_seasonality=True, yearly_seasonality=True):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.model = None
        self.t_min = None
        self.t_max = None
        self.features_columns = []
        
    def _get_seasonality_features(self, dates: pd.Series) -> pd.DataFrame:
        features = pd.DataFrame(index=dates.index)
        day_of_year = dates.dt.dayofyear
        day_of_week = dates.dt.dayofweek
        
        if self.yearly_seasonality:
            for i in range(1, 6):
                features[f"yearly_sin_{i}"] = np.sin(2 * np.pi * i * day_of_year / 365.25)
                features[f"yearly_cos_{i}"] = np.cos(2 * np.pi * i * day_of_year / 365.25)
                
        if self.weekly_seasonality:
            for i in range(1, 4):
                features[f"weekly_sin_{i}"] = np.sin(2 * np.pi * i * day_of_week / 7)
                features[f"weekly_cos_{i}"] = np.cos(2 * np.pi * i * day_of_week / 7)
                
        return features

    def fit(self, df: pd.DataFrame, y: pd.Series, exog: pd.DataFrame = None):
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        self.t_min = df['ds'].min()
        self.t_max = df['ds'].max()
        
        t = (df['ds'] - self.t_min) / (self.t_max - self.t_min) if self.t_max != self.t_min else np.zeros(len(df))
        X = pd.DataFrame({'t': t}, index=df.index)
        
        n_changepoints = min(15, int(0.8 * len(df)))
        if n_changepoints > 0:
            changepoint_t = np.linspace(0.05, 0.8, n_changepoints)
            for idx, cp in enumerate(changepoint_t):
                X[f"cp_{idx}"] = np.maximum(0, t - cp)
                
        S = self._get_seasonality_features(df['ds'])
        X = pd.concat([X, S], axis=1)
        
        if exog is not None:
            X = pd.concat([X, exog], axis=1)
            
        self.model = Ridge(alpha=1.0 / self.changepoint_prior_scale)
        self.model.fit(X, y)
        self.features_columns = list(X.columns)
        return self

    def predict(self, df: pd.DataFrame, exog: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        t = (df['ds'] - self.t_min) / (self.t_max - self.t_min) if self.t_max != self.t_min else np.zeros(len(df))
        X = pd.DataFrame({'t': t}, index=df.index)
        
        n_changepoints = len([c for c in self.features_columns if c.startswith("cp_")])
        if n_changepoints > 0:
            changepoint_t = np.linspace(0.05, 0.8, n_changepoints)
            for idx, cp in enumerate(changepoint_t):
                X[f"cp_{idx}"] = np.maximum(0, t - cp)
                
        S = self._get_seasonality_features(df['ds'])
        X = pd.concat([X, S], axis=1)
        
        if exog is not None:
            X = pd.concat([X, exog], axis=1)
            
        X = X[self.features_columns]
        yhat = self.model.predict(X)
        
        # Simple residuals based std error
        std_err = np.std(yhat) * 0.1
        return pd.DataFrame({
            'ds': df['ds'],
            'yhat': yhat,
            'yhat_lower': yhat - 1.96 * std_err,
            'yhat_upper': yhat + 1.96 * std_err
        }, index=df.index)


def detect_seasonality_and_trend(y: pd.Series) -> tuple[bool, bool]:
    """Uses linear regression for trend detection and seasonal decompose variance ratios for seasonality."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Detrend/Check slope
    x = np.arange(len(y))
    res = OLS(y, add_constant(x)).fit()
    has_trend = res.pvalues[1] < 0.05
    
    # Seasonality check
    has_seasonality = False
    for period in [7, 12, 4]:
        if len(y) > 2 * period:
            try:
                decomp = seasonal_decompose(y, period=period, extrapolate_trend='freq')
                seasonal_var = np.var(decomp.seasonal)
                total_var = np.var(y)
                if total_var > 0 and (seasonal_var / total_var) > 0.08:
                    has_seasonality = True
                    break
            except Exception:
                pass
                
    return has_trend, has_seasonality


def infer_seasonal_period(dates: pd.Series) -> int:
    """Infers seasonal period (s) based on average delta differences."""
    try:
        freq = pd.infer_freq(dates)
        if freq:
            if 'H' in freq:
                return 24
            elif 'D' in freq:
                return 7
            elif 'W' in freq:
                return 52
            elif 'M' in freq:
                return 12
            elif 'Q' in freq:
                return 4
    except Exception:
        pass
        
    # Manual delta calculation
    diffs = dates.diff().dropna()
    if not diffs.empty:
        mean_diff = diffs.mean()
        hours = mean_diff.total_seconds() / 3600.0
        if hours <= 1.5:
            return 24  # hourly
        elif hours <= 25.0:
            return 7   # daily
        elif hours <= 180.0:
            return 4   # weekly
        elif hours <= 750.0:
            return 12  # monthly
            
    return 7  # default fallback


def run_timeseries_pipeline(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
    experiment_id: int,
    business_objective: str = None,
    custom_hyperparameters: dict = None,
    hyperparameter_tuning: bool = False
) -> dict:
    init_mlflow()
    
    stage_durations = {}
    t_start_cleaning = time.time()
    
    # 1. Validation Target Column
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(
            f"Target column '{target_column}' is categorical. "
            "Time-series forecasting requires a numeric target column. Please select a numerical column."
        )
        
    update_progress(experiment_id, "cleaning", "Filtering missing target/date cells and sorting records chronologically.")
    df = df.dropna(subset=[target_column, date_column]).copy()
    if len(df) < 15:
        raise ValueError(f"Dataset has only {len(df)} rows after removing nulls, which is too small for forecasting. Need at least 15 rows.")
        
    df[date_column] = pd.to_datetime(df[date_column], format="mixed")
    df = df.sort_values(date_column).reset_index(drop=True)
    
    # 2. EDA & Stationarity Check
    update_progress(experiment_id, "cleaning", "Checking series stationarity using Augmented Dickey-Fuller test.")
    from statsmodels.tsa.stattools import adfuller
    try:
        adf_val = adfuller(df[target_column].dropna())
        is_stationary = adf_val[1] < 0.05
    except Exception:
        is_stationary = False
        
    stage_durations["cleaning"] = round(time.time() - t_start_cleaning, 4)
    t_start_engineering = time.time()
    
    # 3. Feature Engineering
    update_progress(experiment_id, "engineering", "Engineering lag features, rolling statistics, and date-time components.")
    target_series = df[target_column].astype(float)
    
    # Create copy of dataset to compute features
    df_feat = pd.DataFrame(index=df.index)
    df_feat["lag_1"] = target_series.shift(1)
    df_feat["lag_2"] = target_series.shift(2)
    df_feat["rolling_mean_3"] = target_series.shift(1).rolling(3).mean()
    df_feat["rolling_std_3"] = target_series.shift(1).rolling(3).std()
    
    # Extract Date Components
    df_feat["day"] = df[date_column].dt.day
    df_feat["month"] = df[date_column].dt.month
    df_feat["week"] = df[date_column].dt.isocalendar().week.astype(int)
    df_feat["quarter"] = df[date_column].dt.quarter
    df_feat["year"] = df[date_column].dt.year
    df_feat["dayofweek"] = df[date_column].dt.dayofweek
    
    # Fill remaining NaNs using backfill
    df_exog = df_feat.bfill().ffill()
    
    # 4. Time-based Train/Test Split (80% Train, 20% Holdout)
    split_idx = int(len(df) * 0.8)
    
    y_train = target_series.iloc[:split_idx]
    y_test = target_series.iloc[split_idx:]
    
    exog_train = df_exog.iloc[:split_idx]
    exog_test = df_exog.iloc[split_idx:]
    
    # 5. Algorithm Recommendation
    has_trend, has_seasonality = detect_seasonality_and_trend(y_train)
    
    rec_alg = "arima"
    if has_seasonality:
        rec_alg = "sarima" if len(df) >= 80 else "prophet"
    elif has_trend:
        rec_alg = "arima"
        
    if rec_alg == "arima":
        rec_reason = "ARIMA was recommended because the series exhibits a general trend but lacks distinct seasonality patterns."
    elif rec_alg == "sarima":
        rec_reason = "SARIMA was recommended because the series exhibits strong seasonality and dataset size is sufficient."
    else:
        rec_reason = "Prophet was recommended because the time series contains seasonal patterns and is well suited for modern regression based decomposition."
        
    is_advanced = custom_hyperparameters and "algorithm" in custom_hyperparameters
    selected_alg = custom_hyperparameters["algorithm"] if is_advanced else rec_alg
    
    stage_durations["engineering"] = round(time.time() - t_start_engineering, 4)
    t_start_training = time.time()
    
    # 6. Automatic Hyperparameter Selection & Model Training
    update_progress(experiment_id, "training", f"Training {selected_alg.upper()} forecasting model.")
    leaderboard = []
    best_estimator = None
    best_preds = None
    best_params = {}
    
    # Build models order grids
    d_order = 0 if is_stationary else 1
    s_period = infer_seasonal_period(df[date_column])
    
    with mlflow.start_run(run_name=f"experiment-{experiment_id}") as run:
        mlflow.log_param("workflow_type", "time_series")
        mlflow.log_param("business_objective", str(business_objective))
        mlflow.log_param("recommended_algorithm", rec_alg)
        mlflow.log_param("selected_algorithm", selected_alg)
        
        # Grid Search / Evaluation loop for algorithms
        run_prophet = (not is_advanced) or (selected_alg == "prophet")
        run_arima = (not is_advanced) or (selected_alg == "arima")
        run_sarima = (not is_advanced) or (selected_alg == "sarima")

        # ---- PROPHET ----
        prophet_rmse, prophet_mae, prophet_mape = None, None, None
        prophet_model = None
        prophet_preds = []
        prophet_train_time = 0.0
        if run_prophet:
            update_progress(experiment_id, "training", "Fitting prophet estimator on training fold.")
            try:
                prior_scale = 0.05
                if is_advanced and selected_alg == "prophet":
                    prior_scale = float(custom_hyperparameters.get("changepoint_prior_scale", 0.05))
                    
                t_start = time.time()
                # Prepare prophet df (needs ds, y)
                prophet_df_train = pd.DataFrame({"ds": df[date_column].iloc[:split_idx], "y": y_train})
                pm = ProphetRegression(changepoint_prior_scale=prior_scale, weekly_seasonality=has_seasonality, yearly_seasonality=True)
                pm.fit(prophet_df_train, y_train, exog=exog_train)
                prophet_train_time = time.time() - t_start
                
                # Predict
                prophet_df_test = pd.DataFrame({"ds": df[date_column].iloc[split_idx:]})
                pred_df = pm.predict(prophet_df_test, exog=exog_test)
                prophet_preds = pred_df["yhat"].values
                
                prophet_rmse = float(np.sqrt(mean_squared_error(y_test, prophet_preds)))
                prophet_mae = float(mean_absolute_error(y_test, prophet_preds))
                prophet_mape = calculate_mape(y_test, prophet_preds)
                prophet_model = pm
            except Exception as e:
                print(f"Prophet training failed: {e}")
            
        # ---- ARIMA ----
        arima_rmse, arima_mae, arima_mape = None, None, None
        arima_fit = None
        arima_preds = []
        arima_train_time = 0.0
        arima_best_order = (1, d_order, 1)
        if run_arima:
            update_progress(experiment_id, "training", "Fitting arima estimator on training fold.")
            try:
                from statsmodels.tsa.arima.model import ARIMA
                if is_advanced and selected_alg == "arima":
                    p = int(custom_hyperparameters.get("p", 1))
                    d = int(custom_hyperparameters.get("d", d_order))
                    q = int(custom_hyperparameters.get("q", 1))
                    arima_best_order = (p, d, q)
                elif hyperparameter_tuning:
                    # Sweep order
                    best_aic = float("inf")
                    for p_val in [0, 1, 2]:
                        for q_val in [0, 1, 2]:
                            try:
                                # Use simple ARIMA on target (no exog) with cov_type='none' and maxiter=20 for hyperparameter tuning
                                am = ARIMA(y_train, order=(p_val, d_order, q_val))
                                fit_am = am.fit(method_kwargs={'maxiter': 20}, cov_type='none')
                                if fit_am.aic < best_aic:
                                    best_aic = fit_am.aic
                                    arima_best_order = (p_val, d_order, q_val)
                            except Exception:
                                pass
                else:
                    arima_best_order = (1, d_order, 1)
                                
                t_start = time.time()
                am = ARIMA(y_train, order=arima_best_order)
                arima_fit = am.fit(method_kwargs={'maxiter': 30}, cov_type='none')
                arima_train_time = time.time() - t_start
                
                # Predict
                arima_preds = arima_fit.forecast(steps=len(y_test))
                if hasattr(arima_preds, "values"):
                    arima_preds = arima_preds.values
                    
                arima_rmse = float(np.sqrt(mean_squared_error(y_test, arima_preds)))
                arima_mae = float(mean_absolute_error(y_test, arima_preds))
                arima_mape = calculate_mape(y_test, arima_preds)
            except Exception as e:
                print(f"ARIMA training failed: {e}")
            
        # ---- SARIMA ----
        sarima_rmse, sarima_mae, sarima_mape = None, None, None
        sarima_fit = None
        sarima_preds = []
        sarima_train_time = 0.0
        sarima_best_order = (1, d_order, 1)
        sarima_seasonal_order = (0, 1, 0, s_period)
        if run_sarima:
            update_progress(experiment_id, "training", "Fitting sarima estimator on training fold.")
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                if is_advanced and selected_alg == "sarima":
                    p = int(custom_hyperparameters.get("p", 1))
                    d = int(custom_hyperparameters.get("d", d_order))
                    q = int(custom_hyperparameters.get("q", 1))
                    P = int(custom_hyperparameters.get("P", 0))
                    D = int(custom_hyperparameters.get("D", 1))
                    Q = int(custom_hyperparameters.get("Q", 0))
                    s = int(custom_hyperparameters.get("s", s_period))
                    sarima_best_order = (p, d, q)
                    sarima_seasonal_order = (P, D, Q, s)
                elif hyperparameter_tuning:
                    # Sweep basic seasonal orders
                    best_aic = float("inf")
                    for p_val in [0, 1]:
                        for q_val in [0, 1]:
                            try:
                                sm = SARIMAX(
                                    y_train, 
                                    order=(p_val, d_order, q_val), 
                                    seasonal_order=(0, 1, 0, s_period)
                                )
                                fit_sm = sm.fit(maxiter=20, cov_type='none', disp=False)
                                if fit_sm.aic < best_aic:
                                    best_aic = fit_sm.aic
                                    sarima_best_order = (p_val, d_order, q_val)
                            except Exception:
                                pass
                else:
                    sarima_best_order = (1, d_order, 1)
                    sarima_seasonal_order = (0, 1, 0, s_period)
                                
                t_start = time.time()
                sm = SARIMAX(y_train, order=sarima_best_order, seasonal_order=sarima_seasonal_order)
                sarima_fit = sm.fit(maxiter=30, cov_type='none', disp=False)
                sarima_train_time = time.time() - t_start
                
                # Predict
                sarima_preds = sarima_fit.forecast(steps=len(y_test))
                if hasattr(sarima_preds, "values"):
                    sarima_preds = sarima_preds.values
                    
                sarima_rmse = float(np.sqrt(mean_squared_error(y_test, sarima_preds)))
                sarima_mae = float(mean_absolute_error(y_test, sarima_preds))
                sarima_mape = calculate_mape(y_test, sarima_preds)
            except Exception as e:
                print(f"SARIMA training failed: {e}")
            
        # Assemble leaderboard
        if arima_rmse is not None:
            leaderboard.append({
                "model_name": "arima",
                "metrics": {"rmse": arima_rmse, "mae": arima_mae, "mape": arima_mape},
                "params": {"order": arima_best_order},
                "estimator": arima_fit,
                "preds": arima_preds,
                "train_time": round(arima_train_time, 4)
            })
        if sarima_rmse is not None:
            leaderboard.append({
                "model_name": "sarima",
                "metrics": {"rmse": sarima_rmse, "mae": sarima_mae, "mape": sarima_mape},
                "params": {"order": sarima_best_order, "seasonal_order": sarima_seasonal_order},
                "estimator": sarima_fit,
                "preds": sarima_preds,
                "train_time": round(sarima_train_time, 4)
            })
        if prophet_rmse is not None:
            leaderboard.append({
                "model_name": "prophet",
                "metrics": {"rmse": prophet_rmse, "mae": prophet_mae, "mape": prophet_mape},
                "params": {"changepoint_prior_scale": prior_scale},
                "estimator": prophet_model,
                "preds": prophet_preds,
                "train_time": round(prophet_train_time, 4)
            })
            
        # Rank and pick champion
        if not leaderboard:
            raise ValueError("All forecasting models failed to train.")
            
        leaderboard = sorted(leaderboard, key=lambda x: x["metrics"]["rmse"])
        champion = leaderboard[0]
        
        best_model_name = champion["model_name"]
        best_estimator = champion["estimator"]
        best_preds = champion["preds"]
        best_params = champion["params"]
        best_metrics = champion["metrics"]
        
        stage_durations["training"] = round(time.time() - t_start_training, 4)
        t_start_evaluation = time.time()

        # 7. Generate Forecast Plots (Matplotlib actual vs forecast and residual subplots)
        update_progress(experiment_id, "evaluation", "Compiling results and generating forecast timeline charts.")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Actual vs Forecast
        axes[0].plot(df[date_column].iloc[:split_idx], y_train, label="Train Actual", color="#337ab7")
        axes[0].plot(df[date_column].iloc[split_idx:], y_test, label="Test Actual", color="#5cb85c")
        axes[0].plot(df[date_column].iloc[split_idx:], best_preds, label="Forecasted", color="#d9534f", linestyle="--")
        axes[0].set_title("Actual vs Forecasted Timeline")
        axes[0].set_xlabel("Timeline")
        axes[0].set_ylabel("Target Value")
        axes[0].legend()
        
        # Residual plot
        residuals = np.array(y_test) - np.array(best_preds)
        axes[1].scatter(df[date_column].iloc[split_idx:], residuals, color="#6A5ACD", alpha=0.7)
        axes[1].axhline(y=0, color="gray", linestyle="--")
        axes[1].set_title("Residual Plot")
        axes[1].set_xlabel("Timeline")
        axes[1].set_ylabel("Error")
        
        plt.tight_layout()
        plot_filename = f"forecast_{uuid.uuid4().hex}.png"
        plot_path = os.path.join(REPORTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        
        # 8. Business Insights Generation (Statistics-driven)
        insights = []
        slope = (best_preds[-1] - best_preds[0]) / len(best_preds) if len(best_preds) > 1 else 0
        trend_label = "increasing" if slope > 0.05 else "declining" if slope < -0.05 else "stable"
        
        insights.append(f"Trend Analysis: Forecasted timeline shows a general {trend_label} pattern.")
        if has_seasonality:
            insights.append("Seasonality Peaks: Regular repeating cyclical patterns were detected in historical observations.")
        else:
            insights.append("Seasonality Peaks: No strong repeating patterns detected in sequence cycles.")
            
        if business_objective in ("sales_forecasting", "revenue_forecasting"):
            insights.append(f"Business Alert: Projected revenue/sales are expected to remain {trend_label} over the next steps.")
        elif business_objective in ("demand_forecasting", "inventory_forecasting"):
            insights.append(f"Business Alert: Inventory requirements are expected to {'rise' if slope > 0.05 else 'drop' if slope < -0.05 else 'remain steady'}.")
        else:
            insights.append(f"Insights: Expected target value fluctuation is estimated around {round(np.std(best_preds), 2)} units.")
            
        insights.append(f"Model Summary: Selected {best_model_name.upper()} model (MAPE: {round(best_metrics['mape'], 2)}%).")
        
        # Log to MLflow
        mlflow.log_metric("rmse", best_metrics["rmse"])
        mlflow.log_metric("mae", best_metrics["mae"])
        mlflow.log_metric("mape", best_metrics["mape"])
        
        # Save model joblib artifact
        model_id = uuid.uuid4().hex
        model_path = os.path.join(MODELS_DIR, f"model_{model_id}.joblib")
        joblib.dump({
            "model": best_estimator,
            "feature_configuration": list(df_exog.columns),
            "hyperparameters": best_params,
            "metrics": best_metrics,
            "forecast_results": best_preds.tolist(),
            "timestamp": dt.datetime.now().isoformat(),
            "task": "prophet" if best_model_name == "prophet" else "statsmodels",
            "target_column": target_column,
            "date_column": date_column
        }, model_path)
        
        stage_durations["evaluation"] = round(time.time() - t_start_evaluation, 4)
        update_progress(experiment_id, "completed", "Time Series forecasting model generated successfully.")
        
        # Leaderboard compile
        formatted_leaderboard = []
        for idx, item in enumerate(leaderboard):
            formatted_leaderboard.append({
                "rank": idx + 1,
                "model_name": item["model_name"],
                "metrics": item["metrics"],
                "train_time": item.get("train_time", 0.05),
                "inference_time": item.get("inference_time", 0.005)
            })
            
        return {
            "mlflow_run_id": run.info.run_id,
            "best_model_name": best_model_name,
            "metrics": {item["model_name"]: item["metrics"] for item in leaderboard},
            "best_metrics": best_metrics,
            "leaderboard": formatted_leaderboard,
            "business_insights": insights,
            "model_path": model_path,
            "shap_plot_path": plot_path,
            "task": "time_series",
            "actual": y_test.tolist(),
            "predicted": best_preds.tolist(),
            "dates": df[date_column].iloc[split_idx:].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "recommendation_reason": rec_reason,
            "stage_durations": stage_durations
        }
