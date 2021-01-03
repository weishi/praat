import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import collections


def LoadFormantData():
    df = pd.read_csv(input_base_dir / 'normalized.csv')
    return df


def WriteLangWordCsv(df, output_dir, target_lang, target_word):
    matched_rows = []
    for _, row in df.iterrows():
        comps = row['Filename'].split('_')
        lang = comps[0]
        word = comps[3]
        pos = comps[4]
        if lang == target_lang and int(word) == target_word and pos == 'b':
          matched_rows.append(row)
    mdf = pd.DataFrame(matched_rows)
    output_df = mdf[['Filename', 'Annotation', 'breakF1', 'breakF2', 'deltaF1', 'deltaF2', 'delta_barkF1', 'delta_barkF2']]
    filename = target_lang + '_a_' + str(target_word) + '_b'
    output_df_csv = output_dir / (filename + '.CSV')
    print(output_df_csv)
    output_df.to_csv(output_df_csv, index=False)


input_base_dir = Path('./analysis/item_a/output/')
output_base_dir = Path('./analysis/output_per_word/')
shutil.rmtree(output_base_dir, ignore_errors=True)
output_base_dir.mkdir(parents=True, exist_ok=True)

df_formant = LoadFormantData()
LANGS = ['S', 'M', 'B']
WORDS = [1,4,5,7,8,9,10,11,14]
for lang in LANGS:
  for word in WORDS:
    WriteLangWordCsv(df_formant, output_base_dir, lang, word)