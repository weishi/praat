import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import collections


def LoadFormantData():
    df = pd.read_csv(input_base_dir / 'normalized.csv')
    return df


def GetLabel(row):
    comps = row['Filename'].split('_')
    lang = comps[0]
    pos = comps[4]
    label = lang + pos + '=' + row['Annotation']
    return label


def WriteGroupCsv(df, output_dir, label_cond, word_cond, filename):
    person_word_label = {}
    for _, row in df.iterrows():
        comps = row['Filename'].split('_')
        lang = comps[0]
        if lang.startswith('norm'):
          continue
        label = GetLabel(row)
        person = comps[2]
        word = comps[3]
        if person in person_word_label:
          if word in person_word_label[person]:
            person_word_label[person][word].add(label)
          else:
            person_word_label[person][word] = {label}
        else:
          person_word_label[person] = {word: {label}}
    person_matched_words = collections.defaultdict(set)
    for person, value in person_word_label.items():
        for word, label in value.items():
          matched=True
          for l in label_cond:
            if l not in label: 
              matched=False
              break
          if matched:
            person_matched_words[person].add(word)
    matched_rows = []
    for _, row in df.iterrows():
        comps = row['Filename'].split('_')
        word = comps[3]
        person = comps[2]
        if GetLabel(row) not in word_cond:
            continue
        if person in person_matched_words:
            if word in person_matched_words[person]:
                matched_rows.append(row)
    mdf = pd.DataFrame(matched_rows)
    output_df_csv = output_dir / (filename + '.CSV')
    print(output_df_csv)
    mdf.to_csv(output_df_csv, index=False)


input_base_dir = Path('./item_a/output/')
output_base_dir = Path('./item_a/output_group/')
shutil.rmtree(output_base_dir, ignore_errors=True)
output_base_dir.mkdir(parents=True, exist_ok=True)

df_formant = LoadFormantData()
output_df = []
# delta
WriteGroupCsv(df_formant, output_base_dir, ['Sa=a1', 'Sb=a1'], ['Sa=a1'], 'delta_Sa=a1@Sb=a1__Sa@a1')
WriteGroupCsv(df_formant, output_base_dir, ['Sa=a1', 'Sb=a1'], ['Sb=a1'], 'delta_Sa=a1@Sb=a1__Sb@a1')
WriteGroupCsv(df_formant, output_base_dir, ['Sa=a1', 'Sb=a2'], ['Sa=a1'], 'delta_Sa=a1@Sb=a2__Sa@a1')
WriteGroupCsv(df_formant, output_base_dir, ['Sa=a1', 'Sb=a2'], ['Sb=a2'], 'delta_Sa=a1@Sb=a2__Sb@a2')
WriteGroupCsv(df_formant, output_base_dir, ['Sb=a1'], ['Mb=a2'], 'delta_Sb=a1__Mb@a2')
WriteGroupCsv(df_formant, output_base_dir, ['Sb=a2'], ['Mb=a2'], 'delta_Sb=a2__Mb@a2')
WriteGroupCsv(df_formant, output_base_dir, [], ['Sb=a2'], 'delta_all__Sb@a2')
WriteGroupCsv(df_formant, output_base_dir, [], ['Mb=a2', 'Bb=a2'], 'delta_all__Mb@a2_Bb@a2')

# break
WriteGroupCsv(df_formant, output_base_dir, [], ['Sb=a2'], 'break_all__Sb@a2')
WriteGroupCsv(df_formant, output_base_dir, ['Sb=a1'], ['Mb=a2'], 'break_Sb=a1__Mb@a2')
WriteGroupCsv(df_formant, output_base_dir, ['Sb=a2'], ['Mb=a2'], 'break_Sb=a2__Mb@a2')
WriteGroupCsv(df_formant, output_base_dir, [], ['Mb=a2', 'Bb=a2'], 'break_all__Mb@a2_Bb@a2')