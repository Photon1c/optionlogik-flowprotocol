#from sim_generator import generate_mock_chain
from price_predictor import predict_price_change
from data_loader import load_option_chain_data

actual_close_price = 186.00  # Last actual close for 07/28

#Mock Data
# Assume last predicted price from 07-28 was 201.00
#starting_price = 186
reflexivity_weight = 0.043  # Best from 07-28

# Generate tomorrow's option chain
#tomorrow_strikes = list(range(125, 220, 5))
#tomorrow_chain = generate_mock_chain(tomorrow_strikes, date="2025-07-29")
option_df = load_option_chain_data("hsy", date="07_28_2025")
predicted_price = predict_price_change(option_df, actual_close_price, reflexivity_weight)

print(f"🔮 Adjusted Prediction for 2025-07-29: {predicted_price:.2f}")

#Mock Data
# Predict next day's price
#next_day_prediction = predict_price_change(tomorrow_chain, starting_price, reflexivity_weight)

#print(f"📈 Predicted price for 2025-07-29: {next_day_prediction:.2f}")
