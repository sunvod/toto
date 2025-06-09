#!/usr/bin/env python3
"""
Wrapper script to run CSV testing with proper environment setup.
"""

import os
import sys
import subprocess

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
toto_path = os.path.join(project_root, "toto")

# Add paths to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, toto_path)

# Set environment variables
os.environ["PYTHONPATH"] = f"{project_root}:{toto_path}:{os.environ.get('PYTHONPATH', '')}"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Import and run the test
from test_csv_data import main

if __name__ == "__main__":
    # Pass command line arguments to main
    sys.argv[0] = "test_csv_data.py"  # Fix the script name
    main()