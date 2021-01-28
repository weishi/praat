# Runs after normalization and per_person_ratio_and_factor
import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import collections


def LoadFormantData():
    df = pd.read_csv(input_base_dir / 'normalized.csv')
    return df

def GetPerson(row):
  comps = row.name.split('_')
  return int(comps[2])

def GetPersonFromFilename(row):
  comps = row['Filename'].split('_')
  return int(comps[2])

def GetPersonWord(row):
  comps = row['Filename'].split('_')
  key = comps[1] + '_' + comps[2] + '_' + comps[3]
  return key

def GetPos(row):
  comps = row['Filename'].split('_')
  return comps[4]

def GetAnnotation(row):
  if row.name.endswith('b_02'):
    return 'a2'
  else:
    return 'a1'

def GetSlice(row, a2_words):
  comps = row['Filename'].split('_')
  pos = comps[4]
  # S_a_01
  lang_person = comps[0] + '_' + comps[1] + '_' + comps[2]
  key = GetPersonWord(row)
  category = '01'
  if key in a2_words:
    category = '02'
  slice_key = lang_person + '_' + pos + '_' + category
  return slice_key
  
def GetIsSba2(row, a2_words):
  key = GetPersonWord(row)
  if key in a2_words:
    return 'Yes'
  else:
    return 'No'

def AggregateFourGroups(df_formant, df_sf, output_dir):
    matched_rows = []
    a2_words = set()
    for _, row in df_formant.iterrows():
      if not row['Filename'].startswith('S') and not row['Filename'].startswith('M'):
        continue
      matched_rows.append(row)
      if row['Annotation'] == 'a2':
        # a_09_01
        key = GetPersonWord(row)
        a2_words.add(key)
    df = pd.DataFrame(matched_rows)
    df['SaSbSlice'] = df.apply (GetSlice, a2_words=a2_words, axis=1)
    df['IsSba2'] = df.apply(GetIsSba2, a2_words=a2_words, axis=1)
    df['Pos'] = df.apply(GetPos, axis=1)
    df['Person'] = df.apply(GetPersonFromFilename, axis=1)
    df = df.merge(df_sf, on='Person')
    df.to_csv(output_dir / 'all_plot_raw_data.CSV', index=False)
    

    grouped_df = df.groupby(['SaSbSlice'])['breakF1', 'breakF2', 'deltaF1','deltaF2','delta_barkF1','delta_barkF2'].mean()
    grouped_df['Filename'] = grouped_df.index
    grouped_df['Annotation'] = grouped_df.apply(GetAnnotation, axis=1)
    grouped_df['Person'] = grouped_df.apply(GetPerson, axis=1)
    output_df = grouped_df.merge(df_sf, on='Person')
    print(grouped_df.columns)
    print(df_sf)
    print(output_df)
    col_order = ['Filename', 'Annotation', 
                 'sb_chg_m', 'sb_a2', 'sb_a1_a2', 'sb_a2_div_a1_a2',
                 'Gender','AgeGroup', 'Year', 'Age','Family1','Family2','Family3','Family4','Education1','Career1','Career2','Language1', 
                 'breakF1', 'breakF2', 'deltaF1', 'deltaF2', 'delta_barkF1', 'delta_barkF2']
    output_df[col_order].to_csv(output_dir / 'all_plot_grouped_data.CSV', index=False)

    s_a_all = output_df[output_df['Filename'].str.contains(r'S_a_\d\d_b_\d\d$', regex=True)]
    s_a_all[col_order].to_csv(output_dir / 'S_a_all@Sb.CSV', index=False)
    m_a_all = output_df[output_df['Filename'].str.contains(r'M_a_\d\d_b_\d\d$', regex=True)]
    m_a_all[col_order].to_csv(output_dir / 'M_a_all@M.CSV', index=False)

    return df

input_base_dir = Path('./analysis/item_a_sm/output/')
output_base_dir = Path('./analysis/item_a_sm/output/')
#shutil.rmtree(output_base_dir, ignore_errors=True)
#output_base_dir.mkdir(parents=True, exist_ok=True)

df_formant = pd.read_csv(input_base_dir / 'normalized.csv')
df_chg_sf = pd.read_csv(input_base_dir / 'change_ratio_with_sf.csv')

AggregateFourGroups(df_formant, df_chg_sf, output_base_dir)
