import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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


def risk_metrics(returns, risk_free_rate=0.0, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) and annualized Sharpe Ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns of portfolio/asset.
    risk_free_rate : float
        Daily risk-free rate (default=0).
    confidence_level : float
        Confidence level for VaR (default=0.95).
    
    Returns:
    --------
    dict : VaR and Sharpe ratio
    """
    # Historical VaR
    var_hist = np.percentile(returns, (1 - confidence_level) * 100)
    
    # Parametric VaR (Normal distribution assumption)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    z_score = np.abs(np.percentile(np.random.randn(100000), (1 - confidence_level) * 100))
    var_param = mean_ret - z_score * std_ret
    
    # Sharpe Ratio (annualized)
    excess_daily = returns - risk_free_rate
    sharpe_daily = excess_daily.mean() / excess_daily.std()
    sharpe_annual = sharpe_daily * np.sqrt(252)
    
    return {
        "Historical VaR": var_hist,
        "Parametric VaR": var_param,
        "Annualized Sharpe Ratio": sharpe_annual
    }

def plot_risk_metrics(returns, file_path, confidence_level=0.95):
    """
    Visualize returns distribution with VaR threshold.
    """
    var_hist = np.percentile(returns, (1 - confidence_level) * 100)
    
    plt.figure(figsize=(10,6))
    sns.histplot(returns, bins=50, kde=True, color='blue')
    plt.axvline(var_hist, color='red', linestyle='--', label=f'VaR {int(confidence_level*100)}%')
    plt.title("Portfolio Returns Distribution with VaR")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.legend()

    if file_path:
        plt.savefig(file_path)
    plt.show()