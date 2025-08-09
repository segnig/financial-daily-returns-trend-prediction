from statsmodels.tsa.stattools import adfuller

def adf_test(series, title='Series'):
    """
    Run Augmented Dickey-Fuller test and print result.
    Returns dict with statistic, pvalue, usedlag, nobs, critical values.
    """
    res = adfuller(series.dropna(), autolag='AIC')
    out = {
        'adf_stat': res[0],
        'p_value': res[1],
        'used_lag': res[2],
        'nobs': res[3],
        'critical_values': res[4],
        'icbest': res[5] if len(res) > 5 else None
    }
    print(f'ADF Test for {title}')
    print(f'  ADF Statistic: {out["adf_stat"]:.4f}')
    print(f'  p-value: {out["p_value"]:.4f}')
    for k, v in out['critical_values'].items():
        print(f'    Critical {k}: {v:.4f}')
    if out['p_value'] < 0.05:
        print(" -> Reject H0: series is likely stationary.")
    else:
        print(" -> Fail to reject H0: series is likely non-stationary.")
    return out


def descriptive_stats(df, price_col='Close', return_col='daily_return'):
    """Return dict with describe for price and returns + skew/kurtosis."""
    out = {}
    out['price_describe'] = df[price_col].describe()
    out['returns_describe'] = df[return_col].describe()
    out['returns_skew'] = df[return_col].skew()
    out['returns_kurtosis'] = df[return_col].kurtosis()  # excess kurtosis by pandas
    return out