from pathlib import Path
import pandas as pd


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def CheckFormant():
    invalid_rows = []
    for input in sorted(input_base_dir.rglob('allformants*')):
        # output_csv = input.parent / (input.stem + '_new.CSV')
        # with open(input, 'r') as inf, open(output_csv, 'w') as of:
        #     for line in inf:
        #         trim = [field.strip() for field in line.split(',')]
        #         of.write(','.join(trim)+'\n')

        single_df = pd.read_csv(input)
        single_df.drop(single_df.filter(regex="Unname"), axis=1, inplace=True)

        # assert single_df.shape[1] == 181
        matched_rows = []
        for _, row in single_df.iterrows():
            cols1 = ['F1_' + str(i) for i in range(2, 11)]
            cols2 = ['F2_' + str(i) for i in range(2, 11)]
            cols = cols1+cols2
            has_undefined = False
            for col in cols:
                if 'undefined' in str(row[col]):
                    has_undefined = True
            if has_undefined:
                bad_row = str(input) + ' ' + row['Filename']
                invalid_rows.append(bad_row)
                print(bad_row)
                continue
            if str(row['Annotation']) == 'nan':
              continue
            matched_rows.append(row)
        mdf = pd.DataFrame(matched_rows)
        output_df = pd.concat(
            [mdf[['Filename']],
             mdf[['Annotation']],
             mdf.loc[:, mdf.columns.str.startswith("F1")],
             mdf.loc[:, mdf.columns.str.startswith("F2")],
             ], axis=1)
        output_df_csv = input.parent / (input.stem + '_trimmed.CSV')
        output_df.to_csv(output_df_csv, index=False)
    return invalid_rows

def CheckHnr():
    invalid_rows = []
    for input in sorted(input_base_dir.rglob('*_HNR.txt')):
        output_csv = input.parent / (input.stem + '_new.txt')
        with open(input, 'r') as inf, open(output_csv, 'w') as of:
            for line in inf:
                trim = [field.strip() for field in line.split('\t')]
                trim[0] = rchop(trim[0], '.wav')
                of.write(','.join(trim)+'\n')

        single_df = pd.read_csv(output_csv)
        single_df.drop(single_df.filter(regex="Unname"), axis=1, inplace=True)
        assert single_df.shape[1] == 16
        for _, row in single_df.iterrows():
            has_undefined = False
            for col in single_df:
                if 'undefined' in str(row[col]):
                    has_undefined = True
            if has_undefined:
                bad_row = str(input) + ' ' + row['Filename']
                invalid_rows.append(bad_row)
                print(bad_row)
    return invalid_rows

input_base_dir = Path('./testall/')
output = []
output = output + CheckFormant()
# output = output + CheckHnr()
with open('report.txt', 'w') as of:
  for row in output:
    of.write(row+'\n')
