import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar


def removeChars(s):
    for c in [' ', '\\', '/', '^']:
        s = s.replace(c, '')
    return s


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

def IsValid(row):
  comps = row['Filename'].split('_')
  assert len(comps) == 5 or len(comps) == 6
  lang = comps[0]
  if lang.startswith('norm'):
    return True
  if int(comps[3]) == 13:
    return False
  if lang.startswith('S'):
    return 'S'+comps[4]+'='+row['Annotation'] in ('Sa=a1', 'Sb=a1', 'Sb=a2')
  if lang.startswith('M') or lang.startswith('B'):
    return row['Annotation'] == 'a2'
  print(row)
  raise NotImplementedError

def LoadFormantData():
    all_data = []
    for input in sorted(input_base_dir.glob('*.CSV')):
        print(input)
        single_df = pd.read_csv(input, converters={
            'Annotation': removeChars}, na_values=['--undefined--', 'null'], skipinitialspace=True, sep="\s*[,]\s*", engine='python')
        single_df.drop(single_df.filter(regex="Unname"), axis=1, inplace=True)
        all_data.append(single_df)
    df = pd.concat(all_data, ignore_index=True)
    df = df[df.Annotation != 'null']
    # df.drop_duplicates(subset=['Filename', 'Annotation'], keep='first', inplace=True)
    print('Num files', len(all_data))
    print('Final', df.shape)
    return df


def GetBreak(row, formant):
  x = np.arange(0, 19)
  y = row[formant+'_2': formant+'_20'].to_numpy(dtype='float')
  coeff = np.polyfit(x, y, 4)
  line = np.poly1d(coeff)
  linedd = np.polyder(line, 2)
  linedd_max = minimize_scalar(-linedd, bounds=(0, 19), method='bounded')
  return linedd_max.x / 19.0

def ComputeBreak(df, output_dir):
  df['breakF1'] = df.apply (lambda row: GetBreak(row,'F1'), axis=1)
  df['breakF2'] = df.apply (lambda row: GetBreak(row,'F2'), axis=1)
  plt.figure(figsize=(10, 10))
  plt.scatter(
    df[df['Annotation'] == 'b1']['breakF1'], 
    df[df['Annotation'] == 'b1']['breakF2'], 
    s=4, c='r', label='S=b1')
  plt.scatter(
    df[df['Annotation'] == 'b2']['breakF1'], 
    df[df['Annotation'] == 'b2']['breakF2'], 
    s=4, c='g', label='S=b2')
  plt.title("Scatter plot for item_b breakF1/F2")
  plt.xlabel('breakF1')
  plt.ylabel('breakF2')
  plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
  plt.savefig(output_dir / 'item_b_scatter_break.png', bbox_inches="tight")
  plt.close()
  return df

def ComputeDelta(df, output_dir):
  df['deltaF1'] = df['F1_6'] - df['F1_16']
  df['deltaF2'] = df['F2_6'] - df['F2_16']
  plt.figure(figsize=(10, 10))
  plt.scatter(
    df[df['Annotation'] == 'b1']['deltaF1'], 
    df[df['Annotation'] == 'b1']['deltaF2'], 
    s=4, c='r', label='S=b1')
  plt.scatter(
    df[df['Annotation'] == 'b2']['deltaF1'], 
    df[df['Annotation'] == 'b2']['deltaF2'], 
    s=4, c='g', label='S=b2')
  plt.title("Scatter plot for item_b deltaF1/F2")
  plt.xlabel('deltaF1')
  plt.ylabel('deltaF2')
  plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
  plt.savefig(output_dir / 'item_b_scatter_delta.png', bbox_inches="tight")
  plt.close()
  return df

input_base_dir = Path('./analysis/item_b/')
output_base_dir = Path('./analysis/output_item_b/')
shutil.rmtree(output_base_dir, ignore_errors=True)
output_base_dir.mkdir(parents=True, exist_ok=True)

df_formant = LoadFormantData()
df_formant = ComputeBreak(df_formant, output_base_dir)
df_formant = ComputeDelta(df_formant, output_base_dir)
output_df = pd.concat(
            [df_formant[['Filename']],
             df_formant[['Annotation']],
             df_formant[['breakF1']],
             df_formant[['breakF2']],
             df_formant[['deltaF1']],
             df_formant[['deltaF2']],
             ], axis=1)
output_df.to_csv(output_base_dir / 'item_b_break_delta.csv', index=False)


