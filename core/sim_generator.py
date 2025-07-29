import pandas as pd
import numpy as np
from parameter_loader import get_base_symbol, get_expiration_date, get_mock_data_config

def generate_mock_chain(strikes, date=None, base_symbol=None):
    # Use parameters if not provided
    if date is None:
        date = get_expiration_date()
    if base_symbol is None:
        base_symbol = get_base_symbol()
    data = []
    for strike in strikes:
        call_symbol = f"{base_symbol}C{int(strike * 1000):08d}"
        put_symbol = f"{base_symbol}P{int(strike * 1000):08d}"
        call_iv = round(np.random.uniform(1.5, 4.0), 4)
        put_iv = round(np.random.uniform(1.5, 3.0), 4)

        row = {
            "Expiration Date": date,
            "Calls": call_symbol,
            "Last Sale": round(np.random.uniform(50, 100), 2),
            "Net": round(np.random.uniform(-5, 5), 3),
            "Bid": round(np.random.uniform(50, 100), 2),
            "Ask": round(np.random.uniform(50, 100), 2),
            "Volume": np.random.randint(0, 1000),
            "IV": call_iv,
            "Delta": round(np.random.uniform(0.95, 1.0), 4),
            "Gamma": round(np.random.uniform(0, 0.001), 4),
            "Open Interest": np.random.randint(0, 500),
            "Strike": strike,
            "Puts": put_symbol,
            "Last Sale_P": 0.01,
            "Net_P": round(np.random.uniform(-0.01, 0.01), 3),
            "Bid_P": 0.00,
            "Ask_P": 0.01,
            "Volume_P": np.random.randint(0, 1000),
            "IV_P": put_iv,
            "Delta_P": round(np.random.uniform(-0.0015, -0.0003), 4),
            "Gamma_P": round(np.random.uniform(0, 0.001), 4),
            "Open Interest_P": np.random.randint(0, 5000)
        }
        data.append(row)

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage using parameters
    from parameter_loader import get_strikes
    
    strikes = get_strikes()
    df = generate_mock_chain(strikes)
    print(f"Generated mock chain for {len(strikes)} strikes: {strikes[:5]}...")
    print(df.head())  # Show a preview
