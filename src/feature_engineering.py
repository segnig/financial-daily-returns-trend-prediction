from scipy import stats
import numpy as np


def add_rolling_stats(df, windows=(7,20,60), column='daily_return'):
    """Add rolling mean/std/var for given windows. Returns dataframe copy."""
    df = df.copy()
    for w in windows:
        df[f'rolling_mean_{w}d'] = df[column].rolling(window=w).mean()
        df[f'rolling_std_{w}d']  = df[column].rolling(window=w).std()
        df[f'rolling_var_{w}d']  = df[column].rolling(window=w).var()
    return df

def detect_outliers_iqr(df, return_col='daily_return', factor=1.5):
    """Return DataFrame of outlier rows using IQR on returns."""
    r = df[return_col].dropna()
    Q1 = r.quantile(0.25)
    Q3 = r.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (r < lower) | (r > upper)
    return df.loc[mask]


def detect_outliers_zscore(df, return_col='daily_return', thresh=3.0):
    z = np.abs(stats.zscore(df[return_col].dropna()))
    mask = z > thresh
    return df.loc[df[return_col].dropna().index[mask]]

def compute_var(series_returns, alpha=0.05, method='historical', horizon_days=1):
    """
    Returns daily VaR at given alpha. For historical method: quantile.
    horizon_days scales using sqrt(horizon_days) for parametric methods.
    """
    if method == 'historical':
        var = -np.percentile(series_returns.dropna(), alpha * 100)
    elif method == 'parametric':  # normal assumption
        mu = series_returns.mean()
        sigma = series_returns.std()
        var = -(mu + sigma * stats.norm.ppf(alpha))
    else:
        raise ValueError("method must be 'historical' or 'parametric'")
    # scale for horizon if needed (simple sqrt rule)
    return var * np.sqrt(horizon_days)

def compute_sharpe(series_returns, risk_free_rate_annual=0.02, periods_per_year=252):
    """
    Returns annualized Sharpe ratio (excess return / volatility).
    series_returns are daily returns (decimal, not %).
    """
    rf_daily = (1 + risk_free_rate_annual) ** (1/periods_per_year) - 1
    excess = series_returns.mean() - rf_daily
    ann_excess = excess * periods_per_year
    ann_vol = series_returns.std() * np.sqrt(periods_per_year)
    if ann_vol == 0:
        return np.nan
    return ann_excess / ann_vol