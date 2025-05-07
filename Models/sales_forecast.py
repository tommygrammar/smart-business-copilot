import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_sales(horizon):
    # 1. Generate synthetic daily sales for 5 years
    np.random.seed(42)
    days = 5 * 365
    date_index = pd.date_range(start='2018-01-01', periods=days, freq='D')
    # Trend: linear increase
    trend = 0.02 * np.arange(days)
    # Weekly seasonality: Monday=1.0,...,Sunday=0.8 pattern
    weekly_pattern = np.tile([1.00, 1.05, 1.10, 1.15, 1.20, 1.10, 0.90], days // 7 + 1)[:days]
    # Annual seasonality: sine wave over 365-day cycle
    annual_pattern = 1 + 0.15 * np.sin(2 * np.pi * np.arange(days) / 365)
    # Base level and noise
    base = 200
    noise = np.random.normal(scale=5, size=days)
    sales = (base + trend) * weekly_pattern * annual_pattern + noise
    data = pd.Series(sales, index=date_index, name='Sales')

    # 2. Automatically detect primary seasonal period via autocorrelation
    acf_vals = acf(data, nlags=365, fft=True)
    # ignore lag 0, find lag with highest autocorrelation
    primary_season = np.argmax(acf_vals[1:]) + 1

    # 3. Fit Holt-Winters with additive trend (damped) and multiplicative seasonality
    model = ExponentialSmoothing(
        data,
        trend='add',
        damped_trend=True,
        seasonal='mul',
        seasonal_periods=primary_season
    )
    fit = model.fit(optimized=True)

    # 4. Forecast next year (365 days)
    
    forecast = fit.forecast(horizon)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
    forecast = pd.Series(forecast.values, index=forecast_index, name='Forecast')

    # 5. Compute 95% bootstrap confidence intervals
    residuals = fit.resid.dropna().values
    n_sims = 1000
    simulations = np.random.choice(residuals, size=(n_sims, horizon), replace=True) + forecast.values
    lower = np.percentile(simulations, 2.5, axis=0)
    upper = np.percentile(simulations, 97.5, axis=0)
    ci_lower = pd.Series(lower, index=forecast_index, name='Lower_95')
    ci_upper = pd.Series(upper, index=forecast_index, name='Upper_95')

    # 6. Combine history, forecast, and intervals
    result = pd.concat([data, forecast, ci_lower, ci_upper], axis=1)

    # 7. Additional insights
    initial = data.iloc[0]
    final = data.iloc[-1]
    growth = (final - initial) / initial * 100
    # Weekly peak day
    weekday_means = data.groupby(data.index.day_name()).mean()
    peak_weekday = weekday_means.idxmax()
    # Annual peak month
    month_means = data.groupby(data.index.month).mean()
    peak_month = pd.to_datetime(f'2025-{month_means.idxmax():02d}-01').strftime('%B')
    avg_forecast = forecast.mean()
    

    # 8. Non-technical summary
    summary = (" # Sales Analysis and Forecast Summary\n\n")
    summary += ("-----------------------------------\n\n")
    summary += (f"Detected primary seasonality of roughly every {primary_season} days.\n\n")
    summary += (f"Over the past 5 years, daily average sales rose from {initial:.1f} to {final:.1f}, a total growth of {growth:.1f}%.\n\n")
    summary += (f"Weekly peaks occur on {peak_weekday}, and annual highs in {peak_month}.\n\n")
    summary += (f"The model forecasts average daily sales of {avg_forecast:.1f} units over the next year.\n\n")
    summary += (f"Forecast uncertainty (95% interval) ranges on average ±{((ci_upper - forecast).mean()):.1f} units per day.\n\n")
    summary += ("Apply these forecasts and intervals to optimize staffing, inventory, and promotions around expected highs and lows.\n\n")

    return summary

