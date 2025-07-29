#!/usr/bin/env python3
"""
Test script to demonstrate data mode switching functionality
"""

from parameter_loader import get_ticker, is_mock_data_mode, should_fallback_to_mock
from data_provider import get_data_info, get_option_chain_data, get_stock_data

def test_data_modes():
    """Test different data modes"""
    print("=== Data Mode Testing ===")
    
    # Get current configuration
    ticker = get_ticker()
    mock_mode = is_mock_data_mode()
    fallback_enabled = should_fallback_to_mock()
    
    print(f"Current Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Mock Mode: {mock_mode}")
    print(f"  Fallback to Mock: {fallback_enabled}")
    print()
    
    # Test data provider
    data_info = get_data_info()
    print(f"Data Provider Info:")
    print(f"  Data Source: {data_info['data_source']}")
    print(f"  Mock Mode: {data_info['mock_mode']}")
    print(f"  Fallback Enabled: {data_info['fallback_enabled']}")
    print()
    
    # Test option chain data
    try:
        print("Testing Option Chain Data...")
        option_df = get_option_chain_data()
        print(f"  Successfully loaded option data with {len(option_df)} rows")
        print(f"  Columns: {list(option_df.columns[:5])}...")
        print(f"  Sample data shape: {option_df.shape}")
    except Exception as e:
        print(f"  Error loading option data: {e}")
    print()
    
    # Test stock data
    try:
        print("Testing Stock Data...")
        stock_df = get_stock_data()
        print(f"  Successfully loaded stock data with {len(stock_df)} rows")
        print(f"  Columns: {list(stock_df.columns)}")
        print(f"  Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
    except Exception as e:
        print(f"  Error loading stock data: {e}")
    print()
    
    print("=== Data Mode Test Complete ===")

def demonstrate_mode_switching():
    """Demonstrate how to switch between modes"""
    print("\n=== Mode Switching Instructions ===")
    print("To switch between mock and real data modes:")
    print()
    print("1. Edit parameters.json:")
    print("   Set 'data_mode.mock_data_mode': true for mock data")
    print("   Set 'data_mode.mock_data_mode': false for real data")
    print()
    print("2. For real data mode, ensure:")
    print("   - Stock CSV files exist in F:/inputs/stocks/")
    print("   - Option CSV files exist in F:/inputs/options/log/")
    print("   - File names match the ticker symbol")
    print()
    print("3. Fallback behavior:")
    print("   - When 'fallback_to_mock': true, system falls back to mock data")
    print("   - When 'fallback_to_mock': false, system raises errors for missing data")
    print()

if __name__ == "__main__":
    test_data_modes()
    demonstrate_mode_switching() 