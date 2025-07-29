#!/usr/bin/env python3
"""
Overnight Prediction System
Automatically trains model, generates predictions, and creates charts with forward projections
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import our modules
from parameter_loader import get_ticker, get_current_price, get_strikes, get_prediction_config, get_visualization_config
from data_provider import get_option_chain_data, get_stock_data, get_data_info
from model_trainer import train_model
from price_predictor import predict_price_change
from error_evaluator import evaluate_prediction

class OvernightPredictor:
    """Automated overnight prediction system"""
    
    def __init__(self):
        self.ticker = get_ticker()
        self.current_price = get_current_price()
        self.strikes = get_strikes()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"prediction_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"�� Starting overnight prediction for {self.ticker}")
        print(f"📁 Output directory: {self.run_dir}")
    
    def train_optimal_model(self):
        """Train the model to find optimal reflexivity weight"""
        print("🧠 Training optimal model...")
        
        # Use recent historical data for training
        try:
            stock_df = get_stock_data()
            if len(stock_df) >= 2:
                # Use last two days for training
                recent_prices = stock_df.tail(2)['Close/Last'].values
                actual_price = recent_prices[-1]
                starting_price = recent_prices[-2]
            else:
                # Fallback to parameter-based training
                actual_price = self.current_price + 2.0  # Simulate 2-point increase
                starting_price = self.current_price
        except Exception as e:
            print(f"⚠️ Using fallback training data: {e}")
            actual_price = self.current_price + 2.0
            starting_price = self.current_price
        
        # Train model
        result = train_model(
            actual_price=actual_price,
            starting_price=starting_price,
            strikes=self.strikes,
            method="mse"
        )
        
        self.best_weight = result['best_weight']
        self.best_error = result['best_error']
        
        print(f"✅ Optimal weight: {self.best_weight:.4f}")
        print(f"✅ Best MSE: {self.best_error:.4f}")
        
        # Save training results
        training_results = {
            "timestamp": self.timestamp,
            "ticker": self.ticker,
            "best_weight": self.best_weight,
            "best_error": self.best_error,
            "training_data": {
                "actual_price": actual_price,
                "starting_price": starting_price,
                "strikes_count": len(self.strikes)
            },
            "error_log": result['log']
        }
        
        with open(self.run_dir / "training_results.json", 'w') as f:
            json.dump(training_results, f, indent=2)
        
        return self.best_weight
    
    def generate_forward_prediction(self):
        """Generate forward prediction for next trading day"""
        print("�� Generating forward prediction...")
        
        # Get current option chain data
        option_df = get_option_chain_data()
        
        # Generate prediction using trained weight
        predicted_price = predict_price_change(
            option_df, 
            starting_price=self.current_price,
            reflexivity_weight=self.best_weight
        )
        
        self.predicted_price = predicted_price
        self.price_change = predicted_price - self.current_price
        self.price_change_pct = (self.price_change / self.current_price) * 100
        
        print(f"📊 Current Price: ${self.current_price:.2f}")
        print(f"🔮 Predicted Price: ${predicted_price:.2f}")
        print(f"📈 Expected Change: ${self.price_change:.2f} ({self.price_change_pct:+.2f}%)")
        
        # Save prediction results
        prediction_results = {
            "timestamp": self.timestamp,
            "ticker": self.ticker,
            "current_price": self.current_price,
            "predicted_price": self.predicted_price,
            "price_change": self.price_change,
            "price_change_pct": self.price_change_pct,
            "model_weight": self.best_weight,
            "prediction_date": datetime.now().strftime("%Y-%m-%d"),
            "next_trading_day": self._get_next_trading_day()
        }
        
        with open(self.run_dir / "prediction_results.json", 'w') as f:
            json.dump(prediction_results, f, indent=2)
        
        return predicted_price
    
    def create_forward_projection_chart(self):
        """Create chart with forward projection"""
        print("📊 Creating forward projection chart...")
        
        # Get historical data for context
        try:
            stock_df = get_stock_data()
            historical_prices = stock_df.tail(5)['Close/Last'].values
            historical_dates = stock_df.tail(5)['Date'].dt.strftime('%m-%d').values
        except Exception as e:
            print(f"⚠️ Using mock historical data: {e}")
            # Create mock historical data
            historical_prices = [self.current_price - 2, self.current_price - 1, 
                              self.current_price - 0.5, self.current_price - 0.2, self.current_price]
            historical_dates = ['T-4', 'T-3', 'T-2', 'T-1', 'Today']
        
        # Add prediction to the series
        all_prices = list(historical_prices) + [self.predicted_price]
        all_dates = list(historical_dates) + ['Tomorrow']
        
        # Create the chart
        plt.figure(figsize=(12, 8))
        
        # Plot historical prices
        plt.plot(all_dates[:-1], all_prices[:-1], 'o-', color='blue', 
                linewidth=2, markersize=8, label='Historical Prices')
        
        # Plot prediction
        plt.plot(all_dates[-2:], [all_prices[-2], all_prices[-1]], 'o--', 
                color='red', linewidth=3, markersize=10, label='Forward Projection')
        
        # Highlight the prediction point
        plt.scatter(all_dates[-1], all_prices[-1], color='red', s=200, 
                   zorder=5, edgecolors='black', linewidth=2)
        
        # Add price labels
        for i, (date, price) in enumerate(zip(all_dates, all_prices)):
            if i == len(all_dates) - 1:  # Prediction point
                plt.annotate(f'${price:.2f}', (date, price), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                           fontweight='bold', color='white')
            else:
                plt.annotate(f'${price:.2f}', (date, price), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9)
        
        # Customize chart
        plt.title(f'{self.ticker} Forward Price Projection\n'
                 f'Predicted Change: ${self.price_change:.2f} ({self.price_change_pct:+.2f}%)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add prediction details
        prediction_text = (f'Model Weight: {self.best_weight:.4f}\n'
                         f'Prediction Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n'
                         f'Next Trading Day: {self._get_next_trading_day()}')
        plt.figtext(0.02, 0.02, prediction_text, fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.run_dir / f"{self.ticker}_forward_projection_{self.timestamp}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Chart saved: {chart_path}")
        return chart_path
    
    def create_summary_report(self):
        """Create a summary report"""
        print("📋 Creating summary report...")
        
        report = f"""
# {self.ticker} Overnight Prediction Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Training Results
- Optimal Reflexivity Weight: {self.best_weight:.4f}
- Best MSE: {self.best_error:.4f}
- Training Data Points: {len(self.strikes)} strikes

## Forward Projection
- Current Price: ${self.current_price:.2f}
- Predicted Price: ${self.predicted_price:.2f}
- Expected Change: ${self.price_change:.2f} ({self.price_change_pct:+.2f}%)

## Prediction Details
- Prediction Date: {datetime.now().strftime("%Y-%m-%d")}
- Next Trading Day: {self._get_next_trading_day()}
- Model Confidence: Based on {len(self.strikes)} option strikes

## Files Generated
- Chart: {self.ticker}_forward_projection_{self.timestamp}.png
- Training Results: training_results.json
- Prediction Results: prediction_results.json

## How to Use This Prediction
1. Check the chart for visual projection
2. Monitor {self.ticker} at market open
3. Compare actual vs predicted price at market close
4. Use results to refine model for future predictions

---
*This prediction is for informational purposes only. Always do your own research.*
"""
        
        report_path = self.run_dir / f"{self.ticker}_prediction_report_{self.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_path}")
        return report_path
    
    def _get_next_trading_day(self):
        """Get the next trading day (simplified - assumes next day)"""
        next_day = datetime.now() + timedelta(days=1)
        return next_day.strftime("%Y-%m-%d")
    
    def run_overnight_prediction(self):
        """Run the complete overnight prediction process"""
        print("�� Starting overnight prediction process...")
        print("=" * 50)
        
        try:
            # Step 1: Train the model
            self.train_optimal_model()
            print()
            
            # Step 2: Generate forward prediction
            self.generate_forward_prediction()
            print()
            
            # Step 3: Create chart
            self.create_forward_projection_chart()
            print()
            
            # Step 4: Create summary report
            self.create_summary_report()
            print()
            
            print("🎉 Overnight prediction complete!")
            print(f"�� All files saved to: {self.run_dir}")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"❌ Error during overnight prediction: {e}")
            return False

def main():
    """Main function to run overnight prediction"""
    predictor = OvernightPredictor()
    success = predictor.run_overnight_prediction()
    
    if success:
        print("✅ Overnight prediction completed successfully!")
        print("📊 Check the output directory for results")
    else:
        print("❌ Overnight prediction failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
