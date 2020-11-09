import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import filter as ft
import groups
import analyzer
import condition


def removeChars(s):
    for c in [' ', '\\', '/', '^']:
        s = s.replace(c, '')
    return s


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

def GetVowel(row):
  comps = row['Filename'].split('_')
  assert len(comps) == 5 or len(comps) == 6
  lang = comps[0]
  if lang.startswith('norm'):
    assert row['Annotation'] in ['a', 'i', 'u']
    return 'norm@' + row['Annotation']
  if lang.startswith('M'):
    assert row['Annotation'] in ['a2']
    return 'M@a2'
  if lang.startswith('B'):
    assert row['Annotation'] in ['a2']
    return 'B@a2'
  row_pos = comps[4]
  assert row_pos in ['a', 'b']
  if row['Annotation'] in [ # 'a', 'b', 'c', 'd', 'e', 
                           'a1', 'a2', 
                           'c1', 'c2', 'c2vs', 'c3', 'c4',
                           'd1', 'd2', 'd3',
                           ]:
    return lang + row_pos + '@' + row['Annotation']
  elif row['Annotation'] in ['d1n', 'd1h']:
    return lang + row_pos + '@' + 'd1'
  elif row['Annotation'] in ['d2n', 'd2h']:
    return lang + row_pos + '@' + 'd2'
  elif row['Annotation'] in ['d3n', 'd3h']:
    return lang + row_pos + '@' + 'd3'
  else:
    print(row)
    raise NotImplementedError

def GetPersonLang(row):
  comps = row['Filename'].split('_')
  return '@'.join(comps[0:3])

def GetGender(row):
  age_gender = int(row['Filename'].split('_')[2])
  if age_gender % 2 == 1:
    return 'M'
  else:
    return 'F'

def GetIsFiRows(row):
  age_gender = int(row['Filename'].split('_')[2])
  # female norm rows are skipped in Fi computation
  if age_gender % 2 == 0:
    return 'No'
  lang = row['Filename'].split('_')[0]
  # replace with norm after real data is ready
  if 'norm' in lang:
    return 'Yes'
  else:
    return 'No'

def LoadFormantData():
    all_data = []

    for input in sorted(input_base_dir.rglob('*.csv')):
        print(input)
        single_df = pd.read_csv(input, converters={
            'Annotation': removeChars}, na_values=['--undefined--'])
        single_df.drop(single_df.filter(regex="Unname"), axis=1, inplace=True)
        clean_df = single_df.dropna(subset=kCols)
        clean_df['Table'] = input
        num_nan = len(single_df) - len(clean_df)
        if num_nan > 0:
            print(input, 'Dropped', num_nan)
        all_data.append(clean_df)
    df = pd.concat(all_data, ignore_index=True)
    df = df[df.Annotation != 'null']
    # df.drop_duplicates(subset=['Filename', 'Annotation'], keep='first', inplace=True)
    df['Vowel'] = df.apply (lambda row: GetVowel(row), axis=1)
    df['PersonLang'] = df.apply (lambda row: GetPersonLang(row), axis=1)
    df['Gender'] = df.apply (lambda row: GetGender(row), axis=1)
    df['IsFiRows'] = df.apply (lambda row: GetIsFiRows(row), axis=1)
    print('Num files', len(all_data))
    print('Final', df.shape)
    output_df = pd.concat(
            [df[['Filename']],
             df[['Annotation']],
             df[['Vowel']],
             df[['PersonLang']],
             df[['Gender']],
             df[['IsFiRows']],
             df[['Table']],
             df[kCols],
             ], axis=1)
    return output_df

def addRnorm(row, R_male, R_female):
  if row['Gender'] == 'M':
    return R_male
  else:
    return R_female

def Normalization(df):
  df['F1_6_u'] = df.groupby('Table')['F1_6'].transform('mean')    
  df['F1_6_std'] = df.groupby('Table')['F1_6'].transform('std')    
  df['F1_6_Z'] = (df['F1_6'] - df['F1_6_u']) / df['F1_6_std']
  print(df)
  # A
  vowel_Fki = ['Sa@a1', 'Sb@a1','Sb@a2', 'M@a2', 'B@a2', 'norm@a', 'norm@i', 'norm@u']
  df_Fki_bar = df.groupby(['Vowel', 'Gender'])['F1_6'].mean()   
  print(df_Fki_bar)
  df_Zki_bar = df.groupby(['Vowel', 'Gender'])['F1_6_Z'].mean()   
  df_Fi_bar = df.groupby(['IsFiRows'])['F1_6'].mean()   
  print(df_Fi_bar)
  df_Zi_bar = df.groupby(['IsFiRows'])['F1_6_Z'].mean()   
  Fi_bar = df_Fi_bar['Yes']
  Zi_bar = df_Zi_bar['Yes']
  sum_F_male = 0
  sum_F_female = 0
  sum_Z_male = 0
  sum_Z_female = 0
  print('Male - Fi_bar:' + str(Fi_bar))
  print('Male - Zi_bar:' + str(Zi_bar))
  for vowel in vowel_Fki:
      try: 
        Fki_bar = df_Fki_bar[vowel]['M']
        print(vowel + ' - Male - Fki_bar:' + str(Fki_bar))
        sum_F_male += abs(Fki_bar - Fi_bar)
      except KeyError:
        print(vowel + ' - Male - No F')
      try: 
        Zki_bar = df_Zki_bar[vowel]['M']
        print(vowel + ' - Male - Zki_bar:' + str(Zki_bar))
        sum_Z_male += abs(Zki_bar - Zi_bar)
      except KeyError:
        print(vowel + ' - Male - No Z')
      try: 
        Fki_bar = df_Fki_bar[vowel]['F']
        print(vowel + ' - Female - Fki_bar:' + str(Fki_bar))
        sum_F_female += abs(Fki_bar - Fi_bar)
      except KeyError:
        print(vowel + ' - Female - No F')
      try: 
        Zki_bar = df_Zki_bar[vowel]['F']
        print(vowel + ' - Female - Zki_bar:' + str(Zki_bar))
        sum_Z_female += abs(Zki_bar - Zi_bar)
      except KeyError:
        print(vowel + ' - Female - No Z')
  print('sum_F_male: %f, sum_Z_male: %f' % (sum_F_male, sum_Z_male))
  print('sum_F_female: %f, sum_Z_female: %f' % (sum_F_female, sum_Z_female))
  R_male = sum_F_male / sum_Z_male
  R_female = sum_F_female / sum_Z_female
  print('R_male: %f, R_female: %f' % (R_male, R_female))
  df['F1_6_R_norm'] = df.apply(addRnorm, args=(R_male, R_female), axis=1)
  df['F1_6_Fi_bar'] = Fi_bar
  df['F1_6_p'] = df['F1_6_Z'] * df['F1_6_R_norm']
  df['F1_6_pp'] = df.groupby(['Table', 'PersonLang'])['F1_6_p'].transform('mean')    
  df['F1_6_norm_final'] = df['F1_6_p'] - (df['F1_6_pp'] - df['F1_6_Fi_bar'])
  return df

def NormalizeColumn(df, col):
  df[col + '_u'] = df.groupby('Table')[col + ''].transform('mean')    
  df[col + '_std'] = df.groupby('Table')[col + ''].transform('std')    
  df[col + '_Z'] = (df[col + ''] - df[col + '_u']) / df[col + '_std']
  print(df)
  # A
  vowel_Fki = ['Sa@a1', 'Sb@a1','Sb@a2', 'M@a2', 'B@a2', 'norm@a', 'norm@i', 'norm@u']
  df_Fki_bar = df.groupby(['Vowel', 'Gender'])[col + ''].mean()   
  print(df_Fki_bar)
  df_Zki_bar = df.groupby(['Vowel', 'Gender'])[col + '_Z'].mean()   
  df_Fi_bar = df.groupby(['IsFiRows'])[col + ''].mean()   
  print(df_Fi_bar)
  df_Zi_bar = df.groupby(['IsFiRows'])[col + '_Z'].mean()   
  Fi_bar = df_Fi_bar['Yes']
  Zi_bar = df_Zi_bar['Yes']
  sum_F_male = 0
  sum_F_female = 0
  sum_Z_male = 0
  sum_Z_female = 0
  print('Male - Fi_bar:' + str(Fi_bar))
  print('Male - Zi_bar:' + str(Zi_bar))
  for vowel in vowel_Fki:
      try: 
        Fki_bar = df_Fki_bar[vowel]['M']
        print(vowel + ' - Male - Fki_bar:' + str(Fki_bar))
        sum_F_male += abs(Fki_bar - Fi_bar)
      except KeyError:
        print(vowel + ' - Male - No F')
      try: 
        Zki_bar = df_Zki_bar[vowel]['M']
        print(vowel + ' - Male - Zki_bar:' + str(Zki_bar))
        sum_Z_male += abs(Zki_bar - Zi_bar)
      except KeyError:
        print(vowel + ' - Male - No Z')
      try: 
        Fki_bar = df_Fki_bar[vowel]['F']
        print(vowel + ' - Female - Fki_bar:' + str(Fki_bar))
        sum_F_female += abs(Fki_bar - Fi_bar)
      except KeyError:
        print(vowel + ' - Female - No F')
      try: 
        Zki_bar = df_Zki_bar[vowel]['F']
        print(vowel + ' - Female - Zki_bar:' + str(Zki_bar))
        sum_Z_female += abs(Zki_bar - Zi_bar)
      except KeyError:
        print(vowel + ' - Female - No Z')
  print('sum_F_male: %f, sum_Z_male: %f' % (sum_F_male, sum_Z_male))
  print('sum_F_female: %f, sum_Z_female: %f' % (sum_F_female, sum_Z_female))
  R_male = sum_F_male / sum_Z_male
  R_female = sum_F_female / sum_Z_female
  print('R_male: %f, R_female: %f' % (R_male, R_female))
  df[col + '_R_norm'] = df.apply(addRnorm, args=(R_male, R_female), axis=1)
  df[col + '_Fi_bar'] = Fi_bar
  df[col + '_p'] = df[col + '_Z'] * df[col + '_R_norm']
  df[col + '_Fijk_bar'] = df.groupby(['Table', 'PersonLang'])[col + '_p'].transform('mean')    
  df[col + '_pp'] = df[col + '_p'] - (df[col + '_Fijk_bar'] - df[col + '_Fi_bar'])
  df[col + '_pp_bark'] = 26.81 / (1 + 1960 / df[col + '_pp']) - 0.53
  return df

def ComputeDelta(df):
  df['deltaF1'] = df['F1_6_pp'] - df['F1_16_pp']
  df['deltaF2'] = df['F2_6_pp'] - df['F2_16_pp']
  df['delta_barkF1'] = df['F1_6_pp_bark'] - df['F1_16_pp_bark']
  df['delta_barkF2'] = df['F2_6_pp_bark'] - df['F2_16_pp_bark']
  return df

input_base_dir = Path('./testall/')
output_base_dir = input_base_dir / 'output/'
shutil.rmtree(output_base_dir, ignore_errors=True)
output_base_dir.mkdir(parents=True, exist_ok=True)

cols1 = ['F1_' + str(i) for i in [6, 16]]
cols2 = ['F2_' + str(i) for i in [6, 16]]
kCols = cols1 + cols2
# kCols = ['F1_6']
df_formant = LoadFormantData()
df_formant = NormalizeColumn(df_formant, 'F1_6')
df_formant = NormalizeColumn(df_formant, 'F1_16')
df_formant = NormalizeColumn(df_formant, 'F2_6')
df_formant = NormalizeColumn(df_formant, 'F2_16')
df_formant = ComputeDelta(df_formant)
df_formant.to_csv(output_base_dir / 'normalized.csv', index=False)

