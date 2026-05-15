import os
from pathlib import Path

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

os.system("kaggle datasets download -d neelghoshal/reddit-mental-health-data -p data/raw --unzip")