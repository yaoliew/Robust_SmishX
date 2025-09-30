#!/bin/bash

# Activate conda environment and run the project
source /home/myid/zl26271/miniconda3/etc/profile.d/conda.sh
conda activate SmishX

echo "SmishX environment activated!"
echo "Available commands:"
echo "  python main.py                    - Run the main evaluation"
echo "  python -c \"from main import *; help(SMSPhishingDetector)\" - Get help"
echo "  python -c \"from main import SMSPhishingDetector; print('Detector class loaded successfully')\" - Test import"

# Keep the environment active
exec bash
