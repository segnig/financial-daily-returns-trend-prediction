"""
visual.py
Visual module for financial daily returns trend prediction project.

This module contains functions for visualizing financial data, including daily returns and trends.
It provides functionality to plot daily returns, trends, rolling statistics, and other financial metrics.
The module is designed to work with pandas DataFrames containing financial time series data.
It supports saving visualizations to files and displaying them interactively in Jupyter notebook environments.
The visualizations help in understanding the underlying trends, volatility, and patterns in financial data.
This module is part of a larger project focused on predicting trends in financial daily returns to optimize portfolio management.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_closing_price(df, ticker, save_path=None):
    """
    Plot the closing price over time.

    Parameters:
    - df (pd.DataFrame): DataFrame with DateTime index and 'Close' column.
    - ticker (str): Stock ticker symbol for the plot title.
    - save_path (str, optional): File path to save the plot image. If None, the plot is only shown.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Closing Price')
    plt.title(f'{ticker} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_daily_returns(df, ticker, save_path=None):
    """
    Plot daily returns as a line plot.

    Parameters:
    - df (pd.DataFrame): DataFrame with DateTime index and 'daily_return' column.
    - ticker (str): Stock ticker symbol for the plot title.
    - save_path (str, optional): File path to save the plot image. If None, the plot is only shown.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['daily_return'], label='Daily Returns', color='orange')
    plt.title(f'{ticker} Daily Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_returns_and_volatility(df, ticker, rolling_window=7, save_path=None):
    """
    Plot daily returns and rolling volatility on the same graph.

    Parameters:
    - df (pd.DataFrame): DataFrame with DateTime index and columns 'daily_return' and 'rolling_std_7d'.
    - ticker (str): Ticker symbol for plot title.
    - rolling_window (int): Window size for rolling calculations (used in label).
    - save_path (str, optional): File path to save the plot image. If None, the plot is just shown.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['daily_return'], label='Daily Return')
    plt.plot(df.index, df['rolling_std_7d'], label=f'{rolling_window}-Day Rolling Volatility', color='red')
    plt.title(f'{ticker} Daily Returns and {rolling_window}-Day Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()
