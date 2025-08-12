import pandas as pd

def generate_forecast(fitted_model, n_periods, forecast_index, alpha=0.05):
    """
    Generates forecasts and confidence intervals from a fitted ARIMA/SARIMA model.

    Args:
        fitted_model (statsmodels.tsa.statespace.sarimax.SARIMAResultsWrapper):
            The fitted model object returned from model.fit().
        n_periods (int): The number of steps ahead to forecast.
        forecast_index (pd.DatetimeIndex): The dates for which the forecast is generated.
                                           This should be the index of the test set.
        alpha (float): The significance level for the confidence intervals (e.g., 0.05 for 95% CI).

    Returns:
        pd.DataFrame: A DataFrame with the forecast, and lower/upper confidence interval bounds,
                      correctly indexed by `forecast_index`.
    """
    print(f"\n--- Generating Forecast for {n_periods} periods ---")

    # Use the get_forecast method which provides predictions and confidence intervals
    forecast_object = fitted_model.get_forecast(steps=n_periods, alpha=alpha)

    # Extract the mean forecast (the predictions)
    mean_forecast = forecast_object.predicted_mean
    
    # Extract the confidence intervals
    conf_int = forecast_object.conf_int()
    
    # Create a clear DataFrame to hold all the information
    forecast_df = pd.DataFrame({
        'forecast': mean_forecast.values,
        'lower_ci': conf_int.iloc[:, 0].values,
        'upper_ci': conf_int.iloc[:, 1].values
    }, index=forecast_index)
    
    print("Forecast generation complete.")
    
    return forecast_df