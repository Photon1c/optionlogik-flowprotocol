import numpy as np
from sim_generator import generate_mock_chain
from price_predictor import predict_price_change
from error_evaluator import evaluate_prediction

def train_model(actual_price, starting_price, strikes, method="mse"):
    best_score = float("inf")
    best_weight = None
    error_log = []

    for weight in np.linspace(0.001, 0.05, 50):  # Try 50 weights from 0.001 to 0.05
        mock_df = generate_mock_chain(strikes)
        predicted = predict_price_change(mock_df, starting_price=starting_price, reflexivity_weight=weight)
        error = evaluate_prediction(predicted, actual_price, method=method)
        error_log.append((weight, predicted, error))

        if error < best_score:
            best_score = error
            best_weight = weight

    return {
        "best_weight": best_weight,
        "best_error": best_score,
        "log": error_log
    }

if __name__ == "__main__":
    actual_price = 201.10
    starting_price = 198.23
    strikes = list(range(125, 220, 5))  # 125 to 215
    result = train_model(actual_price, starting_price, strikes, method="mse")

    print(f"Best Reflexivity Weight: {result['best_weight']:.4f}")
    print(f"Best MSE: {result['best_error']:.4f}")
