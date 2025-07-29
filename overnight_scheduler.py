#!/usr/bin/env python3
"""
Overnight Scheduler
Automatically runs overnight predictions at specified times
"""

import schedule
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_overnight_prediction():
    """Run the overnight prediction script"""
    print(f"�� Starting scheduled overnight prediction at {datetime.now()}")
    
    try:
        # Run the overnight predictor
        result = subprocess.run([sys.executable, "overnight_predictor.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Overnight prediction completed successfully")
            print(result.stdout)
        else:
            print("❌ Overnight prediction failed")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running overnight prediction: {e}")

def setup_schedule():
    """Setup the scheduling for overnight predictions"""
    
    # Schedule for 2:00 AM (after market close, before premarket)
    schedule.every().day.at("02:00").do(run_overnight_prediction)
    
    # Also schedule for 6:00 AM (premarket check)
    schedule.every().day.at("06:00").do(run_overnight_prediction)
    
    print("📅 Scheduled overnight predictions:")
    print("   - 2:00 AM: Post-market analysis")
    print("   - 6:00 AM: Pre-market check")
    print("   - Press Ctrl+C to stop")

def main():
    """Main scheduler function"""
    print("🌙 Overnight Prediction Scheduler")
    print("=" * 40)
    
    setup_schedule()
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped by user")

if __name__ == "__main__":
    main()
