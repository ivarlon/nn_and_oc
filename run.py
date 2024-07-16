# -*- coding: utf-8 -*-
"""
Sets the correct work environment for the code.
"""
import numpy as np
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
sys.path.append(str(root))
sys.path.append(str(root / "utils"))
sys.path.append(str(root / "simple_scalar_problem"))
sys.path.append(str(root / "heat_equation"))

def run_script(script_path):
    with open(script_path) as file:
        exec(file.read(), globals())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description = "Runs a script with the necessary paths set up.")
    parser.add_argument("script", help="The script to run, relative to the root directory")
    
    args = parser.parse_args()
    
    script_path = root / args.script
    run_script(script_path)
