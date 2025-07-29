import pandas as pd
from parameter_loader import get_prediction_config, get_current_price, get_ticker
from data_provider import get_option_chain_data, get_data_info

def predict_price_change(option_chain_df, starting_price=None, reflexivity_weight=None):
    """
    Predicts the next price based on net option volume (call - put) and a reflexivity multiplier.
    """
    # Use parameters if not provided
    if starting_price is None:
        starting_price = get_current_price()
    if reflexivity_weight is None:
        config = get_prediction_config()
        reflexivity_weight = config["reflexivity_weight"]
    
    net_flow = option_chain_df["Volume"].sum() - option_chain_df["Volume_P"].sum()
    price_change = reflexivity_weight * net_flow / 100
    predicted_price = max(0.1, starting_price + price_change)
    return predicted_price


if __name__ == "__main__":
    # Get data info
    data_info = get_data_info()
    print(f"Data Source: {data_info['data_source']} mode for {data_info['ticker']}")
    
    # Get option chain data using the unified provider
    option_df = get_option_chain_data()
    predicted = predict_price_change(option_df)
    print(f"Predicted price for {get_ticker()}: {predicted:.2f}")
