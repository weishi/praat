# Runs after normalization and per_person_ratio_and_factor and pre_plot_aggregation.
import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import collections

def PlotWithSlices(df, data_name, output_dir):
   for group_name in ['Gender', 'AgeGroup', 'Family1', 'Family2', 'Family3', 'Family4', 'Education1', 'Career1', 'Career2', 'Language1']:
      grouped_df = df.groupby([group_name])['deltaF1','deltaF2'].mean()
      grouped_df.to_csv(output_dir / (data_name + '_' + group_name + '_raw.csv'), index=True)
      for formant in ['deltaF1', 'deltaF2']:
        x = []
        y = []
        full_group_name = '@'.join([data_name, formant, group_name])
        for _, row in grouped_df.iterrows():
          x.append(group_name + '=' +str(row.name))
          y.append(row[formant])
        plt.figure(figsize=(10, 6))
        plt.bar(x, y)
        plt.title(full_group_name)
        plt.savefig(output_dir / (full_group_name + '.png'), bbox_inches="tight")
        plt.clf()
        plt.cla()
        plt.close()

def SlicePlotData(df, output_dir):
    matched_rows = []
    sa_a1_sb_a1 = df[df['IsSba2']=='No']
    sa_a1_sb_a1.to_csv(output_dir / 'sa_a1_sb_a1_raw.csv', index=False)
    sa_a1_sb_a1_mean = sa_a1_sb_a1.groupby(['Pos'])['deltaF1', 'deltaF2'].mean()
    sa_a1_sb_a1_mean.to_csv(output_dir / 'sa_a1_sb_a1_mean.csv', index=True)
    sa_a1_sb_a2 = df[df['IsSba2']=='Yes']
    sa_a1_sb_a2.to_csv(output_dir / 'sa_a1_sb_a2_raw.csv', index=False)
    sa_a1_sb_a2_mean = sa_a1_sb_a2.groupby(['Pos'])['deltaF1', 'deltaF2'].mean()
    sa_a1_sb_a2_mean.to_csv(output_dir / 'sa_a1_sb_a2_mean.csv', index=True)

    matched_rows = []
    for _, row in df.iterrows():
      comps = row['Filename'].split('_')
      lang = comps[0]
      pos = comps[4]
      if lang == 'S' and pos == 'b' and row['Annotation'] == 'a2':
        matched_rows.append(row)
    input_df = pd.DataFrame(matched_rows)
    PlotWithSlices(input_df, 'all_s_sb_a2', output_dir)

    matched_rows = []
    for _, row in df.iterrows():
      comps = row['Filename'].split('_')
      lang = comps[0]
      pos = comps[4]
      if lang == 'M' and pos == 'b' and row['Annotation'] == 'a2':
        matched_rows.append(row)
    input_df = pd.DataFrame(matched_rows)
    PlotWithSlices(input_df, 'all_m_mb_a2', output_dir)
 

input_base_dir = Path('./analysis/item_a_sm/output/')
output_base_dir = Path('./analysis/item_a_sm_output/')
shutil.rmtree(output_base_dir, ignore_errors=True)
output_base_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_base_dir / 'all_plot_raw_data.csv')

SlicePlotData(df, output_base_dir)

