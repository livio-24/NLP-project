from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import gzip
import json
import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def parse(path):
    """Parse a gzipped JSON file line by line."""
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)

def getDF(path):
    """Load a JSON file into a pandas DataFrame."""
    df = {}
    for i, d in enumerate(parse(path)):
        df[i] = d
    return pd.DataFrame.from_dict(df, orient='index')

def load_data():
    """Load datasets from specified paths."""
    path = RAW_DATA_DIR / 'Cell_Phones_and_Accessories.json.gz'
    path_meta = RAW_DATA_DIR / 'meta_Cell_Phones_and_Accessories.json.gz'
    
    data = getDF(path)
    meta = getDF(path_meta)
    
    return data, meta




