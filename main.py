"""
Main entry point for Fatigue Detection System
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from gui.main_window import main

if __name__ == "__main__":
    main()
