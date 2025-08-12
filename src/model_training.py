import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import matplotlib.pyplot as plt


import pandas as pd

def split_time_series_data(data, split_date):
    """
    Splits a time series DataFrame into training and testing sets based on a date.

    Args:
        data (pd.DataFrame): The input DataFrame with a datetime index.
        split_date (str): The date to split the data on, in 'YYYY-MM-DD' format.
                          Data up to and including this date will be in the training set.

    Returns:
        tuple: A tuple containing the training DataFrame and the testing DataFrame.
    """
    # Ensure the index is a datetime object
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    # Split the data
    train_data = data.loc[data.index <= split_date]
    test_data = data.loc[data.index > split_date]

    if train_data.empty:
        print("Warning: Training set is empty. Check your split_date.")
    if test_data.empty:
        print("Warning: Test set is empty. Check your split_date.")

    print(f"Data split on: {split_date}")
    print(f"Training set: {train_data.index.min().date()} to {train_data.index.max().date()} ({len(train_data)} rows)")
    print(f"Testing set:  {test_data.index.min().date()} to {test_data.index.max().date()} ({len(test_data)} rows)")

    return train_data, test_data


def train_auto_arima_model(train_data, is_seasonal=False, seasonal_period=12):
    """
    Finds the best ARIMA/SARIMA parameters using auto_arima and trains the model.

    Args:
        train_data (pd.Series or pd.DataFrame): The training time series data.
        is_seasonal (bool): Set to True to search for a SARIMA model, False for ARIMA.
        seasonal_period (int): The number of time steps in a seasonal cycle.
                               Only used if is_seasonal is True.

    Returns:
        tuple: (fitted_model, auto_model) where:
            - fitted_model is the trained statsmodels ARIMA/SARIMAX model
            - auto_model is the pmdarima auto_arima model object
    """
    model_type = "SARIMA" if is_seasonal else "ARIMA"
    print(f"\n--- Starting {model_type} Model Training ---")
    print("Finding optimal parameters with auto_arima...")

    try:
        # Use auto_arima to find the best model order
        auto_model = pm.auto_arima(
            train_data,
            seasonal=is_seasonal,
            m=seasonal_period if is_seasonal else 1,
            stepwise=True,
            suppress_warnings=True,
            trace=True,
            error_action='ignore',
            n_jobs=-1  # Use parallel processing if available
        )

        print("\n--- Best Model Found ---")
        print(auto_model.summary())

        # Train the final model using statsmodels
        print("\nTraining final model with the best parameters...")
        final_model = ARIMA(
            train_data,
            order=auto_model.order,
            seasonal_order=auto_model.seasonal_order if is_seasonal else (0, 0, 0, 0)
        )
        
        fitted_model = final_model.fit()
        print("Model training complete.")
        
        return fitted_model, auto_model

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise