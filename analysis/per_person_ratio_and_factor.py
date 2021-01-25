import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import collections


def ComputeChangeRatio(df, output_dir):
    person_b_count = dict()
    person_b_a1_count = dict()
    person_b_a2_count = dict()

    for _, row in df.iterrows():
      if not row['Filename'].startswith('S'):
        continue
      comps = row['Filename'].split('_')
      person = comps[2]
      pos = comps[4]
      if pos == 'b':
        person_b_count[person] = person_b_count.get(person, 0) + 1
        if row['Annotation'] == 'a1':
          person_b_a1_count[person] = person_b_a1_count.get(person, 0) + 1
        else:
          person_b_a2_count[person] = person_b_a2_count.get(person, 0) + 1
    change_ratio = dict()
    for k,b in person_b_count.items():
      a1 = person_b_a1_count.get(k, 0)
      a2 = person_b_a2_count.get(k, 0)
      change_ratio[int(k)] = (a2-a1)/b
    output_df = pd.DataFrame.from_dict(change_ratio, orient='index')
    output_df.index.name = 'Person'
    output_df.columns = ['sb_chg_m']
    return output_df

input_base_dir = Path('./analysis/item_a_sm/output/')
output_base_dir = Path('./analysis/item_a_sm/output/')
#shutil.rmtree(output_base_dir, ignore_errors=True)
#output_base_dir.mkdir(parents=True, exist_ok=True)

df_formant = pd.read_csv(input_base_dir / 'normalized.csv')
df_sf = pd.read_csv('./analysis/socialfactors.csv')
df_sf = df_sf.set_index('Person')

df_change_ratio = ComputeChangeRatio(df_formant, output_base_dir)
print(df_sf.index)
print(df_change_ratio.index)
joined_df = df_change_ratio.join(df_sf)
joined_df.to_csv(output_base_dir / 'change_ratio_with_sf.CSV', index=True)