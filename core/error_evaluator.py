import pandas as pd

def evaluate_prediction(predicted_price, actual_price, method="mse"):
    """
    Evaluates the prediction accuracy using a specified method.
    """
    if method == "mse":
        return (predicted_price - actual_price) ** 2
    elif method == "mae":
        return abs(predicted_price - actual_price)
    elif method == "percent_error":
        if actual_price == 0:
            return float('inf')
        return abs(predicted_price - actual_price) / actual_price * 100
    else:
        raise ValueError(f"Unsupported method: {method}")

if __name__ == "__main__":
    predicted = 198.23
    actual = 201.10
    print("MSE:", evaluate_prediction(predicted, actual, method="mse"))
    print("MAE:", evaluate_prediction(predicted, actual, method="mae"))
    print("Percent Error:", evaluate_prediction(predicted, actual, method="percent_error"))
