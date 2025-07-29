import pandas as pd
from typing import Dict, Any, Optional
from parameter_loader import get_ticker, is_mock_data_mode, should_fallback_to_mock, get_data_mode_config
from sim_generator import generate_mock_chain
from data_ingestor import ingest_data

class DataProvider:
    """Unified data provider that can switch between mock and real data"""
    
    def __init__(self):
        self.ticker = get_ticker()
        self.mock_mode = is_mock_data_mode()
        self.fallback_to_mock = should_fallback_to_mock()
    
    def get_option_chain_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get option chain data based on configuration.
        
        Args:
            date: Optional date for real data mode
            
        Returns:
            DataFrame with option chain data
        """
        if self.mock_mode:
            return self._get_mock_data()
        else:
            return self._get_real_data(date)
    
    def _get_mock_data(self) -> pd.DataFrame:
        """Generate mock option chain data"""
        from parameter_loader import get_strikes
        
        strikes = get_strikes()
        print(f"Generating mock option chain data for {self.ticker}")
        return generate_mock_chain(strikes)
    
    def _get_real_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """Load real option chain data"""
        try:
            print(f"Loading real option chain data for {self.ticker}")
            data = ingest_data(self.ticker, date=date)
            return data["option_data"]
        except FileNotFoundError as e:
            if self.fallback_to_mock:
                print(f"Real data not found: {e}")
                print("Falling back to mock data...")
                return self._get_mock_data()
            else:
                raise e
    
    def get_stock_data(self) -> pd.DataFrame:
        """
        Get stock data (historical prices).
        
        Returns:
            DataFrame with stock price data
        """
        if self.mock_mode:
            return self._get_mock_stock_data()
        else:
            return self._get_real_stock_data()
    
    def _get_mock_stock_data(self) -> pd.DataFrame:
        """Generate mock stock data"""
        from parameter_loader import get_files_config
        
        files_config = get_files_config()
        historical_file = files_config["historical_prices"]
        
        try:
            return pd.read_csv(historical_file)
        except FileNotFoundError:
            print(f"Mock historical file not found: {historical_file}")
            print("Creating default mock stock data...")
            return self._create_default_mock_stock_data()
    
    def _get_real_stock_data(self) -> pd.DataFrame:
        """Load real stock data"""
        try:
            print(f"Loading real stock data for {self.ticker}")
            data = ingest_data(self.ticker)
            return data["stock_data"]
        except FileNotFoundError as e:
            if self.fallback_to_mock:
                print(f"Real stock data not found: {e}")
                print("Falling back to mock stock data...")
                return self._get_mock_stock_data()
            else:
                raise e
    
    def _create_default_mock_stock_data(self) -> pd.DataFrame:
        """Create default mock stock data if file doesn't exist"""
        from parameter_loader import get_current_price, get_visualization_config
        
        current_price = get_current_price()
        config = get_visualization_config()
        dates = config["dates"]
        
        # Create simple mock data around current price
        data = []
        for i, date in enumerate(dates):
            price = current_price + (i * 0.5)  # Small daily increases
            data.append({
                "Date": date,
                "Open": price - 0.2,
                "High": price + 0.3,
                "Low": price - 0.3,
                "Close/Last": price,
                "Volume": 1000000 + (i * 50000)
            })
        
        return pd.DataFrame(data)
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the current data source"""
        return {
            "ticker": self.ticker,
            "mock_mode": self.mock_mode,
            "fallback_enabled": self.fallback_to_mock,
            "data_source": "mock" if self.mock_mode else "real"
        }

# Global data provider instance
_provider = None

def get_data_provider() -> DataProvider:
    """Get the global data provider instance"""
    global _provider
    if _provider is None:
        _provider = DataProvider()
    return _provider

def get_option_chain_data(date: Optional[str] = None) -> pd.DataFrame:
    """Get option chain data based on configuration"""
    return get_data_provider().get_option_chain_data(date)

def get_stock_data() -> pd.DataFrame:
    """Get stock data based on configuration"""
    return get_data_provider().get_stock_data()

def get_data_info() -> Dict[str, Any]:
    """Get information about the current data source"""
    return get_data_provider().get_data_info() 