from pathlib import Path
import pandas as pd


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def CheckFormant():
    invalid_rows = []
    for input in sorted(input_base_dir.rglob('*_Formant.*')):
        output_csv = input.parent / (input.stem + '_new.CSV')
        with open(input, 'r') as inf, open(output_csv, 'w') as of:
            for line in inf:
                trim = [field.strip() for field in line.split(',')]
                of.write(','.join(trim)+'\n')

        single_df = pd.read_csv(output_csv)
        single_df.drop(single_df.filter(regex="Unname"), axis=1, inplace=True)
        assert single_df.shape[1] == 181
        for _, row in single_df.iterrows():
            cols1 = ['barkF1_' + str(i) for i in range(1, 12)]
            cols2 = ['barkF2_' + str(i) for i in range(1, 12)]
            cols3 = ['barkF3_' + str(i) for i in range(1, 12)]
            cols = cols1+cols2+cols3
            has_undefined = False
            for col in cols:
                if 'undefined' in str(row[col]):
                    has_undefined = True
            if has_undefined:
                bad_row = str(input) + ' ' + row['Filename']
                invalid_rows.append(bad_row)
                print(bad_row)
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

input_base_dir = Path('./test40/')
output = []
output = output + CheckFormant()
output = output + CheckHnr()
with open('report.txt', 'w') as of:
  for row in output:
    of.write(row+'\n')
