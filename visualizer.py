import matplotlib.pyplot as plt
import pandas as pd
from visualizer_utils import get_actual_prices
from parameter_loader import get_visualization_config, get_ticker
from data_provider import get_stock_data, get_data_info

def plot_predictions(dates, actual_prices, predicted_prices, reflexivity_weights=None):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label="Actual Price", color="green", marker='o')
    plt.plot(dates, predicted_prices, label="Predicted Price", color="blue", linestyle="--", marker='x')

    if reflexivity_weights:
        for i, weight in enumerate(reflexivity_weights):
            plt.text(dates[i], predicted_prices[i] + 0.5, f"{weight:.3f}", fontsize=8, ha='center', color='orange')

    plt.title("Reflexive Model: Actual vs Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Get data info
    data_info = get_data_info()
    print(f"Data Source: {data_info['data_source']} mode for {data_info['ticker']}")
    
    # Get configuration from parameters
    config = get_visualization_config()
    dates = config["dates"]
    fallback_prices = config["fallback_prices"]
    predicted_prices = config["predicted_prices"]
    weights = config["reflexivity_weights"]
    
    # Get actual prices from stock data
    try:
        stock_df = get_stock_data()
        # Extract prices for the specified dates
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        actual = []
        for date in dates:
            date_match = stock_df[stock_df['Date'] == pd.to_datetime(date)]
            if not date_match.empty:
                actual.append(float(date_match["Close/Last"].values[0]))
            else:
                actual.append(None)
        
        # Filter out None values and corresponding dates/predictions
        valid_indices = [i for i, price in enumerate(actual) if price is not None]
        if valid_indices:
            actual = [actual[i] for i in valid_indices]
            dates = [dates[i] for i in valid_indices]
            predicted_prices = [predicted_prices[i] for i in valid_indices]
            weights = [weights[i] for i in valid_indices]
            print(f"Loaded actual prices for {get_ticker()}: {actual}")
        else:
            print(f"No valid prices found for {get_ticker()}, using fallback prices")
            actual = fallback_prices
    except Exception as e:
        print(f"Error loading stock data: {e}")
        print(f"Using fallback prices for {get_ticker()}")
        actual = fallback_prices
    
    plot_predictions(dates, actual, predicted_prices, reflexivity_weights=weights)
