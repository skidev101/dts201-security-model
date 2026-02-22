import pandas as pd
import numpy as np
import os


def load_dataset(path):
  print(f"Loading crime dataset from {path}")

  if not os.path.exists(path):
    print(f"Path {path} does not exist")
    return

  # load datasets in chunk for large files
  chunks = []
  for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
    