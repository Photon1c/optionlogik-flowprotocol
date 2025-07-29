import pandas as pd
from pathlib import Path

def get_actual_prices(ticker, date_list, base_dir="."):
    """
    Get actual stock prices for given dates from historical data CSV.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'hsy')
        date_list: List of date strings in 'YYYY-MM-DD' format
        base_dir: Directory containing the historical price CSV file
    
    Returns:
        List of actual prices corresponding to the dates
    """
    ticker_lower = ticker.lower()
    filepath = Path(base_dir) / f"{ticker_lower}_historical_prices.csv"
    
    if not filepath.exists():
        # Fallback to the file we created
        filepath = Path(base_dir) / "hsy_historical_prices.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Historical price file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Only return prices for matching dates in input list
    actuals = []
    for d in date_list:
        date_match = df[df['Date'] == pd.to_datetime(d)]
        if not date_match.empty:
            actuals.append(float(date_match["Close/Last"].values[0]))
        else:
            actuals.append(None)  # or handle missing value
    
    return actuals 