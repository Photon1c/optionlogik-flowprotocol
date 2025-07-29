import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

class ParameterLoader:
    """Utility class to load and manage parameters from parameters.json"""
    
    def __init__(self, config_file: str = "parameters.json"):
        self.config_file = config_file
        self.params = self._load_parameters()
    
    def _load_parameters(self) -> Dict[str, Any]:
        """Load parameters from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_file}: {e}")
    
    def get_ticker(self) -> str:
        """Get the ticker symbol"""
        return self.params["ticker"]["symbol"]
    
    def get_ticker_name(self) -> str:
        """Get the ticker company name"""
        return self.params["ticker"]["name"]
    
    def get_current_price(self) -> float:
        """Get the current stock price"""
        return self.params["ticker"]["current_price"]
    
    def get_price_range(self) -> Dict[str, float]:
        """Get the price range dictionary"""
        return self.params["ticker"]["price_range"]
    
    def get_expiration_date(self) -> str:
        """Get the option expiration date"""
        return self.params["options"]["expiration_date"]
    
    def get_expiration_code(self) -> str:
        """Get the option expiration code"""
        return self.params["options"]["expiration_code"]
    
    def get_strike_range(self) -> Dict[str, float]:
        """Get the strike range configuration"""
        return self.params["options"]["strike_range"]
    
    def get_strikes(self) -> List[float]:
        """Generate the list of strikes based on configuration"""
        strike_config = self.get_strike_range()
        min_strike = strike_config["min"]
        max_strike = strike_config["max"]
        increment = strike_config["increment"]
        
        strikes = []
        current = min_strike
        while current <= max_strike:
            strikes.append(current)
            current += increment
        
        return strikes
    
    def get_symbol_format(self) -> Dict[str, str]:
        """Get the option symbol format configuration"""
        return self.params["options"]["symbol_format"]
    
    def get_mock_data_config(self) -> Dict[str, Any]:
        """Get the mock data generation configuration"""
        return self.params["data_generation"]["mock_data"]
    
    def get_prediction_config(self) -> Dict[str, Any]:
        """Get the prediction configuration"""
        return self.params["prediction"]
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get the visualization configuration"""
        return self.params["visualization"]
    
    def get_files_config(self) -> Dict[str, str]:
        """Get the file configuration"""
        return self.params["files"]
    
    def get_data_loader_config(self) -> Dict[str, Any]:
        """Get the data loader configuration"""
        return self.params["data_loader"]
    
    def get_data_mode_config(self) -> Dict[str, Any]:
        """Get the data mode configuration"""
        return self.params["data_mode"]
    
    def get_base_symbol(self) -> str:
        """Get the base symbol for option generation"""
        ticker = self.get_ticker()
        exp_code = self.get_expiration_code()
        return f"{ticker}{exp_code}"
    
    def reload(self):
        """Reload parameters from file"""
        self.params = self._load_parameters()

# Global parameter loader instance
_loader = None

def get_parameter_loader() -> ParameterLoader:
    """Get the global parameter loader instance"""
    global _loader
    if _loader is None:
        _loader = ParameterLoader()
    return _loader

def get_ticker() -> str:
    """Get the ticker symbol"""
    return get_parameter_loader().get_ticker()

def get_current_price() -> float:
    """Get the current stock price"""
    return get_parameter_loader().get_current_price()

def get_strikes() -> List[float]:
    """Get the list of strikes"""
    return get_parameter_loader().get_strikes()

def get_base_symbol() -> str:
    """Get the base symbol for options"""
    return get_parameter_loader().get_base_symbol()

def get_expiration_date() -> str:
    """Get the expiration date"""
    return get_parameter_loader().get_expiration_date()

def get_prediction_config() -> Dict[str, Any]:
    """Get prediction configuration"""
    return get_parameter_loader().get_prediction_config()

def get_visualization_config() -> Dict[str, Any]:
    """Get visualization configuration"""
    return get_parameter_loader().get_visualization_config()

def get_mock_data_config() -> Dict[str, Any]:
    """Get mock data configuration"""
    return get_parameter_loader().get_mock_data_config()

def get_files_config() -> Dict[str, str]:
    """Get files configuration"""
    return get_parameter_loader().get_files_config()

def get_data_mode_config() -> Dict[str, Any]:
    """Get data mode configuration"""
    return get_parameter_loader().get_data_mode_config()

def is_mock_data_mode() -> bool:
    """Check if mock data mode is enabled"""
    return get_parameter_loader().get_data_mode_config()["mock_data_mode"]

def should_fallback_to_mock() -> bool:
    """Check if system should fallback to mock data when real data is unavailable"""
    return get_parameter_loader().get_data_mode_config()["fallback_to_mock"] 