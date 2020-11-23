import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import os


def GetPerson(row):
  comps = row['Filename'].split('_')
  lang = 'S'
  if comps[0].startswith('B'):
    lang = 'B'
  return lang+comps[2]

input_base_dir = Path('./item_a/output_group')
output_base_dir = Path('./item_a/output_delta_break')
shutil.rmtree(output_base_dir, ignore_errors=True)
output_base_dir.mkdir(parents=True, exist_ok=True)

for input in sorted(input_base_dir.rglob('*.CSV')):
  print(input)
  filename = os.path.basename(input)
  df = pd.read_csv(input)
  df['Person'] = df.apply (lambda row: GetPerson(row), axis=1)
  if filename.startswith('delta'):
    output_df = df.groupby(['Person'])[['deltaF1', 'deltaF2', 'delta_barkF1', 'delta_barkF2']].mean()
    output_df.to_csv(output_base_dir / filename)
  else:
    output_df = df.groupby(['Person'])[['breakF1', 'breakF2']].mean()
    output_df.to_csv(output_base_dir / filename)


