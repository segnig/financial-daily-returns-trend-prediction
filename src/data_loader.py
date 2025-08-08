import yfinance as yf
import pandas as pd
import os

def load_stock_data(ticker, start_date, end_date, save_dir="data/raw"):
    """
    Download historical stock data for a given ticker and date range, save to CSV, and return DataFrame.

    Parameters:
    - ticker (str): Stock symbol, e.g., 'TSLA'
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format
    - save_dir (str): Directory to save CSV file (default: 'data/raw')

    Returns:
    - pandas.DataFrame: Historical stock data
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    # Optional: Check if 'Adj Close' column exists
    if 'Adj Close' not in df.columns:
        print("Warning: 'Adj Close' column not found in downloaded data.")

    csv_path = os.path.join(save_dir, f"{ticker}_historical.csv")
    df.to_csv(csv_path)
    print(f"Data saved to {csv_path}")

    return df



def save_to_csv(df, filename, save_dir="../data/raw"):
    """
    Save DataFrame to CSV file.

    Parameters:
    - df (pandas.DataFrame): DataFrame to save
    - filename (str): Name of the CSV file
    - save_dir (str): Directory to save the CSV file (default: 'data/raw')
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_path = os.path.join(save_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def check_missing_data(df):
    """
    Check for missing data in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame to check

    Returns:
    - pandas.DataFrame: DataFrame with missing data information
    """
    
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if not missing_data.empty:
        print("Missing data found:")
        print(missing_data)
    else:
        print("No missing data found.")
    
    return missing_data
