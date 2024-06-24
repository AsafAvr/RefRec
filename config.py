import os
from pathlib import Path
import logging

class Config:
    BASE_DIR = Path('/home/yandex/DL20232024a/asafavrahamy/RefRec')
    DATA_DIR = BASE_DIR / 'data' / 'ml-1m'
    MODEL_DIR = BASE_DIR / 'RS' / 'model' / 'ml-1m' / 'ctr' / 'DIN'
    DEVICE = 'cuda'


def setup_logging():
    # Configure the logging system
    logging.basicConfig(
        filename='logs/application_improve.log',  # Log file path
        filemode='w',  # Overwrite mode
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        datefmt='%Y-%m-%d %H:%M:%S',  # Timestamp format
        level=logging.INFO  # Logging level
    )
    