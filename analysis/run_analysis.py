import shutil
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import groups


def removeChars(s):
    for c in [' ', '\\', '/', '^']:
        s = s.replace(c, '')
    return s


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def LoadFormantData():
    all_data = []
    for input in sorted(input_base_dir.rglob('*_Formant.*')):
        output_csv = input.parent / (input.stem + '_new.CSV')
        with open(input, 'r') as inf, open(output_csv, 'w') as of:
            for line in inf:
                trim = [field.strip() for field in line.split(',')]
                of.write(','.join(trim)+'\n')

        single_df = pd.read_csv(output_csv, converters={
            'Annotation': removeChars}, na_values=['--undefined--'])
        single_df.drop(single_df.filter(regex="Unname"), axis=1, inplace=True)
        assert single_df.shape[1] == 181
        cols1 = ['barkF1_' + str(i) for i in range(1,12)]
        cols2 = ['barkF2_' + str(i) for i in range(1,12)]
        clean_df = single_df.dropna(subset=cols1+cols2)
        num_nan = len(single_df) - len(clean_df)
        if num_nan > 0:
            print(input, 'Dropped', num_nan)
        all_data.append(clean_df)
    df = pd.concat(all_data, ignore_index=True)
    print('Num files', len(all_data))
    print('Final', df.shape)
    return df


def LoadHnrData():
    all_data = []
    for input in sorted(input_base_dir.rglob('*_HNR.txt')):
        output_csv = input.parent / (input.stem + '_new.txt')
        with open(input, 'r') as inf, open(output_csv, 'w') as of:
            for line in inf:
                trim = [field.strip() for field in line.split('\t')]
                trim[0] = rchop(trim[0], '.wav')
                of.write(','.join(trim)+'\n')

        single_df = pd.read_csv(output_csv, converters={
            'Word': removeChars}, na_values=['--undefined--'])
        single_df.drop(single_df.filter(regex="Unname"), axis=1, inplace=True)
        assert single_df.shape[1] == 16
        clean_df = single_df.dropna()
        num_nan = len(single_df) - len(clean_df)
        if num_nan > 0:
            print(input, 'Dropped', num_nan)
        all_data.append(clean_df)
    df = pd.concat(all_data, ignore_index=True)
    df['Annotation'] = df[['Word']]
    print('Num files', len(all_data))
    print('Final', df.shape)
    return df


def AnalyzeFormant(df, grp):
    group_filters = itertools.product(*grp[0])
    for gf in group_filters:
        group_name = '@'.join([f.GetValue() for f in gf])
        print(group_name)
        matched_rows = []
        for _, row in df.iterrows():
            is_all_matched = [f.IsMatched(row) for f in gf]
            if not np.all(is_all_matched):
                continue
            matched_rows.append(row)
        matched_df = pd.DataFrame(matched_rows)
        if len(matched_rows) == 0:
            print("No data")
            continue
        for analysis in grp[1]:
            if analysis.GetInputType() != "Formant":
                continue
            output_dir = output_base_dir / analysis.GetName()
            output_dir.mkdir(parents=True, exist_ok=True)
            analysis.RunAnalysis(matched_df, group_name, output_dir)


def AnalyzeHnr(df, grp):
    group_filters = itertools.product(*grp[0])
    for gf in group_filters:
        group_name = '@'.join([f.GetValue() for f in gf])
        print(group_name)
        matched_rows = []
        for _, row in df.iterrows():
            is_all_matched = [f.IsMatched(row) for f in gf]
            if not np.all(is_all_matched):
                continue
            matched_rows.append(row)
        if len(matched_rows) == 0:
            print("No data")
            continue
        matched_df = pd.DataFrame(matched_rows)
        for analysis in grp[1]:
            if analysis.GetInputType() != "HNR":
                continue
            output_dir = output_base_dir / analysis.GetName()
            output_dir.mkdir(parents=True, exist_ok=True)
            analysis.RunAnalysis(matched_df, group_name, output_dir)


input_base_dir = Path('./test40/')
output_base_dir = input_base_dir / 'output/'
shutil.rmtree(output_base_dir, ignore_errors=True)

ALL_GROUPS = [groups.GROUP_A, groups.GROUP_C, groups.GROUP_D1, groups.GROUP_D2]
# ALL_GROUPS = [groups.GROUP_C]
for grp in ALL_GROUPS:
    df_formant = LoadFormantData()
    AnalyzeFormant(df_formant, grp)
    df_hnr = LoadHnrData()
    AnalyzeHnr(df_hnr, grp)
