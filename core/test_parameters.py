#!/usr/bin/env python3
"""
Test script to verify parameter system functionality
"""

from parameter_loader import get_ticker, get_current_price, get_strikes, get_base_symbol

def test_parameters():
    print("Testing Parameter System:")
    print(f"Ticker: {get_ticker()}")
    print(f"Current Price: ${get_current_price():.2f}")
    print(f"Base Symbol: {get_base_symbol()}")
    print(f"Number of Strikes: {len(get_strikes())}")
    print(f"First 5 Strikes: {get_strikes()[:5]}")
    print(f"Last 5 Strikes: {get_strikes()[-5:]}")

if __name__ == "__main__":
    test_parameters() 
