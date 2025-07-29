# Generate mock data to work with
import pandas as pd
import numpy as np
from parameter_loader import get_expiration_date, get_mock_data_config, get_ticker, get_base_symbol

# Define function to generate mock option chain data
def generate_mock_option_chain(num_rows=None, expiration_str=None):
    # Use parameters if not provided
    if num_rows is None:
        config = get_mock_data_config()
        num_rows = config["num_rows"]
    if expiration_str is None:
        expiration_str = get_expiration_date()
    
    data = []
    config = get_mock_data_config()
    base_strike = config["base_strike"]
    for i in range(num_rows):
        strike = base_strike + i * 5
        base_symbol = get_base_symbol()
        call_symbol = f"{base_symbol}C{int(strike * 1000):08d}"
        put_symbol = f"{base_symbol}P{int(strike * 1000):08d}"

        call_row = [
            expiration_str,
            call_symbol,
            round(np.random.uniform(50, 100), 2),   # Call Last Sale
            round(np.random.uniform(-5, 5), 3),     # Call Net
            round(np.random.uniform(50, 100), 2),   # Call Bid
            round(np.random.uniform(50, 100), 2),   # Call Ask
            np.random.randint(0, 1000),             # Call Volume
            round(np.random.uniform(1.5, 4.0), 4),  # Call IV
            round(np.random.uniform(0.95, 1.00), 4),# Call Delta
            round(np.random.uniform(0, 0.001), 4),  # Call Gamma
            np.random.randint(0, 500),              # Call OI
            round(strike, 2),                       # Strike
            put_symbol,
            0.01,                                   # Put Last Sale
            round(np.random.uniform(-0.01, 0.01), 3),# Put Net
            0.00,                                   # Put Bid
            0.01,                                   # Put Ask
            np.random.randint(0, 1000),             # Put Volume
            round(np.random.uniform(1.5, 3.0), 4),  # Put IV
            round(np.random.uniform(-0.0015, -0.0003), 4), # Put Delta
            round(np.random.uniform(0, 0.001), 4),  # Put Gamma
            np.random.randint(0, 5000)              # Put OI
        ]
        data.append(call_row)

    columns = [
        "Expiration Date", "Calls", "Last Sale", "Net", "Bid", "Ask", "Volume", "IV", "Delta", "Gamma", "Open Interest",
        "Strike", "Puts", "Last Sale", "Net", "Bid", "Ask", "Volume", "IV", "Delta", "Gamma", "Open Interest"
    ]

    df_mock = pd.DataFrame(data, columns=columns)
    return df_mock

# Generate and display the mock data
mock_option_chain_df = generate_mock_option_chain()
print(f"Generated mock data for {get_ticker()} with {len(mock_option_chain_df)} rows")
print(mock_option_chain_df.head(30))
mock_option_chain_df.to_csv("mock_data.csv", index=False)