#!/usr/bin/env python3
"""
Script to evaluate the SMS Phishing Detector on the augmented dataset.
"""

import sys
import os
from main import evaluate_detector_on_csv

def main():
    # Path to the augmented dataset
    csv_path = "/home/myid/zl26271/robust-smishing/Robust_SmishX/data/augmented_dataset.csv"
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        return
    
    # Number of rows to evaluate (you can adjust this)
    num_rows = 50  # Start with 50 rows for testing
    
    print(f"Evaluating SMS Phishing Detector on {num_rows} rows from {csv_path}")
    print("This may take several minutes as each SMS requires API calls...")
    print("-" * 60)
    
    try:
        # Evaluate the detector
        accuracy = evaluate_detector_on_csv(csv_path, num_rows)
        
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Evaluated {num_rows} rows from the augmented dataset")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Make sure your API keys are properly configured in the environment variables.")

if __name__ == "__main__":
    main()
