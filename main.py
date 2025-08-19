#!/usr/bin/env python3
"""
Simple CLI wrapper for the Satellite Data Analyzer
"""

import sys
import os

# Add current directory to path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sat_data_analyzer import main

if __name__ == "__main__":
    main()