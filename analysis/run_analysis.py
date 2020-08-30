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
        cols1 = ['barkF1_' + str(i) for i in range(1, 12)]
        cols2 = ['barkF2_' + str(i) for i in range(1, 12)]
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

# S_a_01_03_b,a1
# key: a_01_03
# value: S_a_b_a1


def GetUserKeyValue(row):
    comps = row['Filename'].split('_')
    assert len(comps) == 5 or len(comps) == 6
    lang = comps[0]
    vowel = comps[1]
    person = comps[2]
    word = comps[3]
    position = comps[4]
    annotation = row['Annotation']
    key = '_'.join([vowel, person, word])
    value = '_'.join([lang, vowel, position, annotation])
    return key, value


def FilterFormant(df, condition):
    user_map = {}
    for _, row in df.iterrows():
        key, value = GetUserKeyValue(row)
        user_map.setdefault(key, set()).add(value)
    matched_users = set()
    for user, value in user_map.items():
        if condition.IsMatchedUser(value):
            matched_users.add(user)
    matched_rows = []
    for _, row in df.iterrows():
        key, _ = GetUserKeyValue(row)
        if key not in matched_users:
            continue
        matched_rows.append(row)
    print(condition.GetGroupName(), ' = ', len(matched_rows))
    df = pd.DataFrame(matched_rows)
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


def AnalyzeFormantByDemographic(df, grp):
    non_age_gender_grp = []
    for filter_arr in grp[0]:
        has_age_gender = False
        for f in filter_arr:
            if f.GetType() in ('Age', 'Gender'):
                has_age_gender = True
        if not has_age_gender:
            non_age_gender_grp.append(filter_arr)

    group_filters = itertools.product(*non_age_gender_grp)
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
        kAgeFilter = [ft.IsMale(), ft.IsFemale()]
        kGenderFilter = [ft.IsChild(), ft.IsYouth(),
                         ft.IsAdult(), ft.IsSenior()]
        analysis = analyzer.FormantQuantilesByDemographic()
        output_dir = output_base_dir / analysis.GetName()
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis.RunAnalysis(matched_df, kAgeFilter,
                             kGenderFilter, group_name, output_dir)
        analysis.RunAnalysis(matched_df, kGenderFilter,
                             kAgeFilter, group_name, output_dir)


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

CONDITIONS = [
    (condition.Condition(['S_a_a_a1', 'S_a_b_a1']),
     [
        analyzer.FormantQuantilesF1F2SaSb(),
        analyzer.FormantQuantilesF1SbMb(),
        analyzer.FormantQuantilesF2SbMb(),
        analyzer.FormantRegressionSa(),
        analyzer.FormantRegressionSb(),
        analyzer.FormantRegressionMb(),
        analyzer.FormantInflectionMb(),
    ]),
    (condition.Condition(['S_a_a_a1', 'S_a_b_a2']),
     [
        analyzer.FormantQuantilesF1F2SaSb(),
        analyzer.FormantQuantilesF1SbMb(),
        analyzer.FormantQuantilesF2SbMb(),
        analyzer.FormantRegressionSa(),
        analyzer.FormantRegressionSb(),
        analyzer.FormantRegressionMb(),
        analyzer.FormantInflectionMb(),
    ]),
    (condition.Condition(['B_a_b_a2']),
     [
        # analyzer.FormantQuantilesF1F2SaSb(),
        # analyzer.FormantQuantilesF1F2SbMb(),
        # analyzer.FormantQuantilesF1F2MbBb(),
        # analyzer.FormantRegressionSa(),
        # analyzer.FormantRegressionSb(),
        # analyzer.FormantRegressionMb(),
        # analyzer.FormantRegressionBb(),
        # analyzer.FormantInflectionSbMb(),
        # analyzer.FormantInflectionMbBb(),
    ]),
    (condition.Condition(['S_a_a_a1', 'S_a_b_a2']),  # 4
     [
        analyzer.FormantQuantilesF1SbAge(),
        analyzer.FormantQuantilesF2SbAge(),
        analyzer.FormantQuantilesF1MbAge(),
        analyzer.FormantQuantilesF2MbAge(),
        analyzer.FormantRegressionSbAge(),
        analyzer.FormantRegressionMbAge(),
        # analyzer.FormantRegressionBbAge(),
        analyzer.FormantInflectionF1SbAge(),
        analyzer.FormantInflectionF2SbAge(),
        analyzer.FormantInflectionF1MbAge(),
        analyzer.FormantInflectionF2MbAge(),
        # analyzer.FormantInflectionF1BbAge(),
        # analyzer.FormantInflectionF2BbAge(),
    ]),
    (condition.Condition(['S_a_a_a1', 'S_a_b_a2']), # 5
     [
        analyzer.FormantQuantilesF1SbGender(),
        analyzer.FormantQuantilesF2SbGender(),
        analyzer.FormantQuantilesF1MbGender(),
        analyzer.FormantQuantilesF2MbGender(),
        analyzer.FormantRegressionSbGender(),
        analyzer.FormantRegressionMbGender(),
        # analyzer.FormantRegressionBbGender(),
        analyzer.FormantInflectionF1SbGender(),
        analyzer.FormantInflectionF2SbGender(),
        analyzer.FormantInflectionF1MbGender(),
        analyzer.FormantInflectionF2MbGender(),
        # analyzer.FormantInflectionF1BbGender(),
        # analyzer.FormantInflectionF2BbGender(),
    ]),
]

for c, analyzers in CONDITIONS:
    df_formant = LoadFormantData()
    df = FilterFormant(df_formant, c)

    for a in analyzers:
        analyzer_name = a.__class__.__name__
        print(analyzer_name)
        output_dir = output_base_dir / analyzer_name
        output_dir.mkdir(parents=True, exist_ok=True)
        a.RunAnalysis(df, c.GetGroupName(), output_dir)
