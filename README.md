# Option Logik - Flow Protocol

## Advanced Option Chain Data Generation & Prediction System

## 🚀 System Overview

This comprehensive system provides **automated option chain data generation**, **machine learning-based price predictions**, and **overnight forecasting capabilities** for stock market analysis. Originally designed for HSY (Hershey Company), the system now supports multiple tickers with flexible parameter management.

## 🎯 Key Capabilities

### **Core Features**
- ✅ **Multi-Ticker Support**: Switch between any stock (HSY, SPY, AAPL, etc.)
- ✅ **Parameter-Driven Configuration**: Centralized settings via `parameters.json`
- ✅ **Dual Data Modes**: Mock data for testing, real data for production
- ✅ **ML-Powered Predictions**: Automated model training and price forecasting
- ✅ **Overnight Automation**: Scheduled predictions with forward projections
- ✅ **Professional Charts**: High-quality visualizations with predictions
- ✅ **Comprehensive Testing**: Built-in validation and testing tools

## 🔧 System Architecture

### **Core Components**

#### **1. Parameter Management System**
- **`parameters.json`**: Centralized configuration for all system settings
- **`parameter_loader.py`**: Utility module for accessing configuration
- **Dynamic Ticker Switching**: Change ticker with single parameter update
- **Flexible Price Ranges**: Adjust for any stock's trading range

#### **2. Data Generation Engine**
- **`sim_generator.py`**: Main option chain generator with parameter integration
- **`mock_data_creator.py`**: Alternative CSV-based data generation
- **`data_provider.py`**: Unified interface for mock and real data
- **Strike Range Automation**: Dynamic strike generation based on ticker

#### **3. Machine Learning Prediction System**
- **`model_trainer.py`**: Automated model training with optimal weight discovery
- **`price_predictor.py`**: ML-powered price predictions
- **`error_evaluator.py`**: Prediction accuracy assessment
- **Reflexivity Model**: Advanced option flow-based predictions

#### **4. Overnight Prediction System**
- **`overnight_predictor.py`**: Complete automated prediction pipeline
- **Forward Projection Charts**: Professional visualizations with predictions
- **Timestamped Outputs**: Organized results with detailed reports
- **Automated Scheduling**: Ready for overnight automation

## Data Format Consistency

Both scripts now generate option chain data with:
- **Expiration Date**: "Fri Aug 01 2025"
- **Call Symbols**: Format `HSY250801C{strike*1000:08d}` (e.g., HSY250801C00125000)
- **Put Symbols**: Format `HSY250801P{strike*1000:08d}` (e.g., HSY250801P00125000)
- **Strike Range**: 125.00 to 215.00 with 5-point increments

## 📁 File Structure

### **Core System Files**
- `parameters.json`: Centralized configuration for all system parameters
- `parameter_loader.py`: Utility module for loading and accessing parameters
- `data_provider.py`: Unified data provider for switching between mock and real data

### **Data Generation**
- `sim_generator.py`: Main entry point for generating mock option chains
- `mock_data_creator.py`: Alternative data generation script with CSV output
- `data_loader.py`: Real data loading from CSV files
- `data_ingestor.py`: Data ingestion utilities

### **Machine Learning & Prediction**
- `model_trainer.py`: Automated model training with optimal weight discovery
- `price_predictor.py`: ML-powered price predictions
- `error_evaluator.py`: Prediction accuracy assessment
- `overnight_predictor.py`: Complete automated prediction pipeline

### **Visualization & Testing**
- `visualizer.py`: Professional chart generation
- `visualizer_utils.py`: Visualization utilities
- `test_parameters.py`: Parameter system validation
- `test_data_modes.py`: Data mode switching tests

### **Data Files**
- `mock_data.csv`: Generated mock data file
- `sample_historical_data.csv`: Reference data format
- `hsy_historical_prices.csv`: Historical price data for HSY
- `output/`: Directory for overnight prediction results

### **Automation & Scripts**
- `run_overnight_prediction.bat`: Windows batch file for manual runs
- `overnight_scheduler.py`: Automated scheduling (optional)

## 🚀 Usage Guide

### **Quick Start**
```bash
# 1. Configure your ticker in parameters.json
# 2. Test the system
python test_parameters.py
python test_data_modes.py

# 3. Generate predictions
python overnight_predictor.py
```

### **Manual Operations**

#### **Data Generation**
```bash
# Generate mock option chains
python sim_generator.py

# Create CSV mock data
python mock_data_creator.py
```

#### **Prediction & Analysis**
```bash
# Generate price predictions
python price_predictor.py

# Create visualization charts
python visualizer.py

# Run complete overnight prediction
python overnight_predictor.py
```

#### **Testing & Validation**
```bash
# Test parameter system
python test_parameters.py

# Test data mode switching
python test_data_modes.py
```

### **Overnight Automation**

#### **Manual Run (Recommended for Development)**
```bash
# Windows
run_overnight_prediction.bat

# Python
python overnight_predictor.py
```

#### **Scheduled Run (Production)**
```bash
# Start scheduler (requires 'schedule' library)
python overnight_scheduler.py

# Or use Windows Task Scheduler with run_overnight_prediction.bat
```

### **Ticker Switching**
To switch from HSY to SPY (or any other ticker):
1. Edit `parameters.json`
2. Change `"ticker": {"symbol": "SPY", ...}`
3. Update price ranges and strike ranges
4. Run `python test_parameters.py` to verify

## 📊 Technical Specifications

### **Supported Tickers**
- **HSY**: Hershey Company (~$186)
- **SPY**: SPDR S&P 500 ETF (~$635)
- **Any Stock**: Configurable via `parameters.json`

### **Data Specifications**
- **Expiration**: August 1, 2025 (configurable)
- **Strike Ranges**: Dynamic based on ticker price
- **Increment**: 5 points between strikes (configurable)
- **Data Fields**: Complete option chain fields (Last Sale, Net, Bid, Ask, Volume, IV, Delta, Gamma, Open Interest)

### **Machine Learning Model**
- **Algorithm**: Reflexivity-based option flow prediction
- **Training**: Automated weight optimization (0.001 to 0.05 range)
- **Evaluation**: Mean Squared Error (MSE) optimization
- **Features**: Call/Put volume differentials, IV, Delta, Gamma

### **Prediction Capabilities**
- **Time Horizon**: Next trading day
- **Accuracy**: Model-optimized based on historical data
- **Confidence**: Based on option chain analysis
- **Output**: Price prediction with confidence metrics

## ⚙️ Parameter Management System

### **Centralized Configuration**
- **Single Source of Truth**: All settings in `parameters.json`
- **Dynamic Ticker Switching**: Change ticker with one parameter
- **Flexible Price Ranges**: Adjust for any stock's trading range
- **Automated Strike Generation**: Dynamic strike ranges based on price

### **Key Configuration Sections**

#### **Ticker Configuration**
```json
"ticker": {
  "symbol": "SPY",
  "name": "SPDR S&P 500 ETF Trust",
  "current_price": 635.0,
  "price_range": {"min": 625.0, "max": 645.0, "typical": 635.0}
}
```

#### **Option Chain Settings**
```json
"options": {
  "strike_range": {"min": 600.0, "max": 670.0, "increment": 5.0, "count": 15}
}
```

#### **Data Mode Control**
```json
"data_mode": {
  "mock_data_mode": true,
  "fallback_to_mock": true
}
```

#### **Prediction Parameters**
```json
"prediction": {
  "starting_price": 635.0,
  "reflexivity_weight": 0.01
}
```

### **Data Mode System**

#### **Mock Data Mode** (`data_mode.mock_data_mode: true`)
- ✅ **Testing & Development**: Generated mock data for algorithm validation
- ✅ **No Dependencies**: Works without external data files
- ✅ **Consistent Format**: Standardized data structure
- ✅ **Rapid Iteration**: Fast development and testing cycles

#### **Real Data Mode** (`data_mode.mock_data_mode: false`)
- 📊 **Production Analysis**: Uses actual option chain CSV files
- 🔄 **Auto-Discovery**: Automatically finds most recent data
- 🛡️ **Fallback Protection**: Graceful handling of missing data
- 📈 **Market Accuracy**: Real market data for predictions

#### **Fallback Behavior**
- **Smart Fallback**: When `fallback_to_mock: true`, system automatically switches to mock data
- **Error Handling**: When `fallback_to_mock: false`, system raises clear errors
- **Reliability**: Ensures system works in all data scenarios

## 🌙 Overnight Prediction System

### **Automated Pipeline**
1. **🧠 Model Training**: Finds optimal reflexivity weights automatically
2. **🔮 Forward Prediction**: Generates next-day price predictions
3. **📊 Chart Generation**: Creates professional forward projection charts
4. **📁 Organized Output**: Saves everything to timestamped directories

### **Output Structure**
```
output/
└── prediction_20250115_020000/
    ├── SPY_forward_projection_20250115_020000.png  ← Professional chart
    ├── training_results.json                        ← Model training data
    ├── prediction_results.json                      ← Prediction details
    └── SPY_prediction_report_20250115_020000.txt   ← Summary report
```

### **Manual vs Scheduled Runs**
- **Manual Runs**: Immediate execution for testing and development
- **Scheduled Runs**: Automated overnight execution for production
- **Flexible Timing**: Run anytime or set up automated scheduling

## 🎯 Quality Assurance

- ✅ **Multi-Ticker Support**: Works with any stock (HSY, SPY, AAPL, etc.)
- ✅ **Dynamic Configuration**: Automatic strike range adjustment
- ✅ **Professional Output**: High-quality charts and reports
- ✅ **Comprehensive Testing**: Built-in validation tools
- ✅ **Error Handling**: Robust fallback mechanisms
- ✅ **Scalable Architecture**: Easy to extend and modify

## 🔧 Under the Hood: Core Engine Dynamics

### **🏗️ System Architecture Overview**

The system operates on a **modular pipeline architecture** with four core engines working in concert:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Engine   │───▶│  Training Engine│───▶│ Prediction Engine│───▶│ Output Engine   │
│                 │    │                 │    │                 │    │                 │
│ • Mock/Real     │    │ • Weight Search  │    │ • Reflexivity    │    │ • Charts        │
│ • Strike Gen    │    │ • MSE Optimization│   │ • Flow Analysis  │    │ • Reports       │
│ • Format Std    │    │ • Model Selection│   │ • Price Calc     │    │ • Timestamps    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **📊 Data Engine: Foundation Layer**

#### **Option Chain Generation Process**
```python
# Core algorithm in sim_generator.py
def generate_mock_chain(strikes, date, base_symbol):
    for strike in strikes:
        # 1. Calculate option symbols (e.g., HSY250801C00125000)
        call_symbol = f"{base_symbol}C{int(strike * 1000):08d}"
        put_symbol = f"{base_symbol}P{int(strike * 1000):08d}"
        
        # 2. Generate realistic option data
        volume = random.randint(10, 1000)  # Call volume
        volume_put = random.randint(5, 800)  # Put volume
        
        # 3. Calculate Greeks and pricing
        delta = calculate_delta(strike, current_price)
        gamma = calculate_gamma(strike, current_price)
        iv = calculate_implied_volatility(strike, current_price)
```

#### **Data Flow Dynamics**
1. **Parameter Loading**: `parameter_loader.py` reads `parameters.json`
2. **Strike Calculation**: Dynamic range based on current stock price
3. **Symbol Generation**: Standardized format for option symbols
4. **Volume Distribution**: Realistic call/put volume patterns
5. **Greek Calculations**: Delta, Gamma, IV based on moneyness

#### **Data Mode Switching Logic**
```python
# Core logic in data_provider.py
class DataProvider:
    def get_option_chain_data(self):
        if self.mock_mode:
            return self._get_mock_data()  # Generated synthetic data
        else:
            return self._get_real_data()  # Actual market data
```

### **🧠 Training Engine: Machine Learning Core**

#### **Reflexivity Model Fundamentals**
The system uses a **reflexivity-based prediction model** that operates on the principle that option flow can predict price movements:

```python
# Core prediction algorithm in price_predictor.py
def predict_price_change(option_chain_df, starting_price, reflexivity_weight):
    # 1. Calculate net option flow
    net_flow = option_chain_df["Volume"].sum() - option_chain_df["Volume_P"].sum()
    
    # 2. Apply reflexivity weight (learned parameter)
    price_change = reflexivity_weight * net_flow / 100
    
    # 3. Calculate predicted price
    predicted_price = max(0.1, starting_price + price_change)
    return predicted_price
```

#### **Model Training Process**
```python
# Training algorithm in model_trainer.py
def train_model(historical_data, actual_prices):
    best_weight = None
    best_mse = float('inf')
    
    # Grid search through reflexivity weights
    for weight in np.arange(0.001, 0.051, 0.001):
        predictions = []
        for i, data in enumerate(historical_data):
            pred = predict_price_change(data, actual_prices[i], weight)
            predictions.append(pred)
        
        # Calculate Mean Squared Error
        mse = mean_squared_error(actual_prices, predictions)
        
        if mse < best_mse:
            best_mse = mse
            best_weight = weight
    
    return best_weight, best_mse
```

#### **Training Dynamics**
1. **Weight Range**: 0.001 to 0.05 (fine-grained optimization)
2. **Evaluation Metric**: Mean Squared Error (MSE)
3. **Search Strategy**: Grid search with 0.001 increments
4. **Validation**: Historical data backtesting
5. **Optimization**: Finds weight that minimizes prediction error

### **🔮 Prediction Engine: Forward Projection**

#### **Price Prediction Algorithm**
The system implements a **multi-step prediction pipeline**:

```python
# Prediction workflow in overnight_predictor.py
class OvernightPredictor:
    def generate_forward_prediction(self):
        # 1. Get current option chain data
        option_df = get_option_chain_data()
        
        # 2. Apply trained model
        optimal_weight = self.training_results['optimal_weight']
        predicted_price = predict_price_change(option_df, current_price, optimal_weight)
        
        # 3. Calculate confidence metrics
        confidence = self._calculate_prediction_confidence(option_df)
        
        # 4. Generate forward projection
        next_trading_day = self._get_next_trading_day()
        
        return {
            'predicted_price': predicted_price,
            'confidence': confidence,
            'next_trading_day': next_trading_day,
            'model_weight': optimal_weight
        }
```

#### **Confidence Calculation**
```python
def _calculate_prediction_confidence(self, option_df):
    # Based on option flow volume and volatility
    total_volume = option_df["Volume"].sum() + option_df["Volume_P"].sum()
    volatility_factor = option_df["IV"].mean() / 100
    
    # Higher volume + higher volatility = higher confidence
    confidence = min(0.95, (total_volume / 10000) * volatility_factor)
    return confidence
```

### **📈 Output Engine: Visualization & Reporting**

#### **Chart Generation Process**
```python
# Visualization pipeline in overnight_predictor.py
def create_forward_projection_chart(self):
    # 1. Prepare data series
    dates = [current_date, next_trading_day]
    prices = [current_price, predicted_price]
    
    # 2. Create professional chart
    plt.figure(figsize=(12, 8))
    plt.plot(dates, prices, 'b-o', linewidth=2, markersize=8)
    
    # 3. Add prediction annotation
    plt.annotate(f'Predicted: ${predicted_price:.2f}', 
                xy=(next_trading_day, predicted_price),
                xytext=(10, 10), textcoords='offset points')
    
    # 4. Add confidence indicator
    plt.title(f'{ticker} Forward Projection (Confidence: {confidence:.1%})')
```

#### **Report Generation**
```python
def create_summary_report(self):
    report = f"""
    Overnight Prediction Report
    ==========================
    Ticker: {self.ticker}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Model Training Results:
    - Optimal Weight: {self.training_results['optimal_weight']:.4f}
    - Training MSE: {self.training_results['best_mse']:.4f}
    - Training Iterations: {self.training_results['iterations']}
    
    Prediction Results:
    - Current Price: ${self.current_price:.2f}
    - Predicted Price: ${self.prediction_results['predicted_price']:.2f}
    - Price Change: ${price_change:.2f} ({percentage_change:.2f}%)
    - Confidence: {confidence:.1%}
    - Next Trading Day: {next_trading_day}
    """
```

### **🔄 System Integration Dynamics**

#### **Data Flow Pipeline**
```
1. Parameter Loading (parameter_loader.py)
   ↓
2. Data Generation/Retrieval (data_provider.py)
   ↓
3. Model Training (model_trainer.py)
   ↓
4. Price Prediction (price_predictor.py)
   ↓
5. Chart Generation (overnight_predictor.py)
   ↓
6. Report Creation (overnight_predictor.py)
   ↓
7. File Output (output/ directory)
```

#### **Error Handling & Fallback Mechanisms**
```python
# Robust error handling in data_provider.py
def get_option_chain_data(self):
    try:
        if self.mock_mode:
            return self._get_mock_data()
        else:
            return self._get_real_data()
    except Exception as e:
        if self.fallback_to_mock:
            print(f"Falling back to mock data: {e}")
            return self._get_mock_data()
        else:
            raise e
```

#### **Performance Optimization**
- **Caching**: Parameter loader caches configuration
- **Lazy Loading**: Data provider loads data on demand
- **Memory Management**: Efficient DataFrame operations
- **Parallel Processing**: Training can be parallelized (future enhancement)

### **🎯 Key Algorithmic Insights**

#### **Reflexivity Theory Application**
The system implements **George Soros' reflexivity theory** in option markets:
- **Positive Feedback**: High call volume → price increase → more calls
- **Negative Feedback**: High put volume → price decrease → more puts
- **Weight Optimization**: Finds the optimal reflexivity coefficient

#### **Option Flow Analysis**
```python
# Core flow calculation
net_flow = call_volume - put_volume
price_impact = reflexivity_weight * net_flow / volume_normalization_factor
```

#### **Volatility Integration**
The system incorporates implied volatility (IV) in confidence calculations:
- **High IV**: More uncertainty, lower confidence
- **Low IV**: More certainty, higher confidence
- **Volume Weighting**: Higher volume options have more impact

### **🔬 Technical Deep Dive**

#### **Mathematical Foundation**
```
Predicted_Price = Current_Price + (Reflexivity_Weight × Net_Option_Flow / 100)

Where:
- Net_Option_Flow = Σ(Call_Volume) - Σ(Put_Volume)
- Reflexivity_Weight = Optimized via MSE minimization
- Volume normalization = 100 (empirically determined)
```

#### **Model Validation Process**
1. **Historical Backtesting**: Uses past data to validate predictions
2. **Cross-Validation**: Multiple time periods for robustness
3. **Error Analysis**: MSE, MAE, and directional accuracy
4. **Confidence Intervals**: Statistical significance testing

#### **Scalability Considerations**
- **Multi-Ticker Support**: Same algorithm, different parameters
- **Time Series Extension**: Can be extended to multiple days
- **Feature Engineering**: Additional factors can be incorporated
- **Model Ensemble**: Multiple prediction methods can be combined
