import unittest
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    test_file = os.path.join("tests", "test_constrained_min.py")
    if os.path.exists(test_file):
        result = unittest.main(module="tests.test_constrained_min", exit=False)
    else:
        print(f"Error: Test file {test_file} does not exist.")
