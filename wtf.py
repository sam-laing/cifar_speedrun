import sys
print("Python path:", sys.path)
print("Current directory:", __file__)

try:
    from utils.model import CifarNet
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    
    # Try to see what's in the file
    import os
    if os.path.exists("utils/model.py"):
        with open("utils/model.py", "r") as f:
            print("File contents:", f.read())
    else:
        print("File doesn't exist!")