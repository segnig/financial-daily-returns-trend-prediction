# Time Series Forecasting for Portfolio Management Optimization

## Project Overview

This project is a response to the Week 11 Challenge for the 10 Academy: Artificial Intelligence Mastery program. The goal is to assist "Guide Me in Finance (GMF) Investments," a financial advisory firm, by applying time series forecasting and portfolio optimization techniques to enhance their investment strategies.

The core idea is to leverage historical financial data to forecast the performance of a key asset (Tesla - TSLA) and then use this forecast, combined with historical data from other assets (BND and SPY), to construct an optimal investment portfolio based on Modern Portfolio Theory (MPT). The project culminates in backtesting the proposed strategy to evaluate its potential effectiveness.

## Business Objective

GMF Investments aims to use data-driven insights to provide clients with tailored investment strategies that maximize returns while minimizing risk. This project demonstrates a practical workflow for achieving this by:
1.  Forecasting future market trends for a high-growth, high-risk asset.
2.  Optimizing a multi-asset portfolio based on these forecasts.
3.  Validating the strategy through historical simulation (backtesting).

This project acknowledges the **Efficient Market Hypothesis (EMH)**, using forecasting not as a perfect price predictor, but as a valuable input into a broader decision-making framework for portfolio management.

## Data

The analysis uses historical daily price data for three distinct assets, sourced from Yahoo Finance (`yfinance`) for the period of **July 1, 2015, to July 31, 2025**.

### Assets:
*   **Tesla (TSLA):** A high-growth, high-volatility stock from the consumer discretionary sector.
*   **Vanguard Total Bond Market ETF (BND):** A stable, low-risk bond ETF that provides income and stability.
*   **S&P 500 ETF (SPY):** A diversified ETF that tracks the S&P 500 index, offering broad exposure to the U.S. market.

Each dataset includes the standard OHLC (Open, High, Low, Close) prices, Adjusted Close, and Volume.

## Methodology

The project is structured into five key tasks:

### Task 1: Data Preprocessing and Exploratory Data Analysis (EDA)
-   **Data Loading:** Fetched historical data for TSLA, BND, and SPY using the `yfinance` library.
-   **Data Cleaning:** Handled missing values using forward-fill (`ffill`) to ensure time series continuity.
-   **EDA:**
    -   Visualized closing prices to identify long-term trends.
    -   Calculated and plotted daily returns to analyze volatility.
    -   Used rolling means and standard deviations to observe short-term trends and volatility shifts.
-   **Stationarity Testing:** Performed the Augmented Dickey-Fuller (ADF) test on both closing prices and daily returns to check for stationarity, a key assumption for ARIMA models.

### Task 2: Time Series Forecasting Models
-   **Objective:** Forecast the future stock price of TSLA.
-   **Models Implemented:**
    1.  **ARIMA (AutoRegressive Integrated Moving Average):** A classical statistical model. The optimal `(p, d, q)` parameters were found using `pmdarima.auto_arima`.
    2.  **LSTM (Long Short-Term Memory):** A deep learning model (a type of RNN) capable of learning complex, non-linear patterns in sequential data.
-   **Evaluation:** The models were trained on data from 2015-2023 and evaluated on data from 2024-2025 using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). The LSTM model was selected as the superior model for forecasting.

### Task 3: Forecast Future Market Trends
-   **Forecasting:** The trained LSTM model was used to generate a 12-month forecast for TSLA's stock price.
-   **Analysis:** The forecast was visualized with confidence intervals.
-   **Interpretation:** The analysis focused on the predicted trend, the implications of the widening confidence intervals (increasing uncertainty over time), and the potential market opportunities and risks identified from the forecast.

### Task 4: Portfolio Optimization
-   **Objective:** Construct an optimal portfolio using the principles of Modern Portfolio Theory (MPT).
-   **Inputs:**
    -   **Expected Returns:** The LSTM model's forecast was used for TSLA's expected return. Historical average returns were used for BND and SPY.
    -   **Covariance Matrix:** Calculated from the historical daily returns of all three assets.
-   **Optimization:**
    -   The `PyPortfolioOpt` library was used to generate the **Efficient Frontier**, which visualizes the set of optimal portfolios.
    -   Two key portfolios were identified on the frontier: the **Maximum Sharpe Ratio Portfolio** and the **Minimum Volatility Portfolio**.
-   **Recommendation:** Based on the analysis, an optimal portfolio allocation was recommended, along with its expected annual return, volatility, and Sharpe ratio.

### Task 5: Strategy Backtesting
-   **Objective:** Validate the proposed strategy by simulating its performance on historical data.
-   **Process:**
    -   A **backtesting period** was defined (August 1, 2024 - July 31, 2025).
    -   A **benchmark portfolio** (e.g., 60% SPY / 40% BND) was established for comparison.
    -   The performance of the recommended optimal portfolio was simulated over the backtesting period.
-   **Analysis:** The cumulative returns of the strategy and the benchmark were plotted and compared. Final performance metrics (total return, Sharpe ratio) were calculated to determine if the data-driven strategy added value.

## Technologies and Libraries

-   **Language:** Python
-   **Data Handling & Analysis:** pandas, numpy
-   **Data Retrieval:** yfinance
-   **Statistical Modeling:** statsmodels, pmdarima
-   **Deep Learning:** tensorflow, keras, scikit-learn (for scaling)
-   **Portfolio Optimization:** PyPortfolioOpt
-   **Visualization:** matplotlib, seaborn

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/segnig/financial-daily-returns-trend-prediction.git
    cd financial-daily-returns-trend-prediction
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
     ```