import importlib
import sys

# Ensure configuration is reloaded every time it runs
if 'src.config' in sys.modules:
    importlib.reload(sys.modules['src.config'])
    # Reload related modules
    if 'src.main' in sys.modules:
        importlib.reload(sys.modules['src.main'])
    if 'src.models' in sys.modules:
        importlib.reload(sys.modules['src.models'])

from src.main import main

if __name__ == "__main__":
    main() 