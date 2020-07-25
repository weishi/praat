import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar


def fitLine(row, formantName, start, end, outputDir):
    key = '@'.join([row['Filename'], row['Annotation'], formantName])
    x = np.arange(2, 11)
    y = row[formantName + '_' +
            str(start): formantName + '_' + str(end)].to_numpy(dtype='float')
    coeff = np.polyfit(x, y, 4)
    line1 = np.poly1d(coeff)
    line1d = np.polyder(line1, 1)
    line1dd = np.polyder(line1, 2)
    line1dd_max = minimize_scalar(-line1dd, bounds=(2, 10), method='bounded')
    inflection = line1dd_max.x

    plt.plot(x, y, 'o')
    plt.plot(x, line1(x), label='fitted line')
    plt.plot(x, line1d(x), label='1st deriv')
    plt.plot(x, line1dd(x), label='2nd deriv')
    plt.axvline(x=inflection, linestyle='dashed', label='inflection')
    plt.legend(loc='best')
    plt.title(key)
    # plt.show()
    plt.savefig(outputDir / (key + '.png'))
    plt.clf()
    plt.cla()
    # return pd.Series(coeff, index=['x4', 'x3', 'x2', 'x1', 'x0'])
    return pd.Series(inflection, index=['Inflection_'+formantName])


def removeChars(s):
    for c in [' ', '\\', '/', '^']:
        s = s.replace(c, '')
    return s


def processSingleFormant(input, outputDir):
    df = pd.read_csv(input, converters={'Annotation': removeChars})
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    coeffs_f1 = df.apply(fitLine, axis=1,
                         formantName='barkF1', start=2, end=10, outputDir=outputDir)
    coeffs_f2 = df.apply(fitLine, axis=1,
                         formantName='barkF2', start=2, end=10, outputDir=outputDir)
    df = pd.concat([df, coeffs_f1, coeffs_f2], axis=1)
    df.to_csv(outputDir/input.name, index=False)

#input = Path('S/formants/a/01_Formant.CSV')
#outputDir = Path('output/S/formants/a/01_Formant')
# processFormant(input, outputDir)


def processAllFormant():
    for input in sorted(Path('./').rglob('*Formant.CSV')):
        outputDir = 'output' / input.parent / input.stem
        print(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)
        processSingleFormant(input, outputDir)


# shutil.rmtree(Path('output'), ignore_errors=True)
# processAllFormant()

class Analyzer(object):
    def RunAnalysis(self, df, group_name, output_base_dir):
        raise NotImplementedError

    def GetName(self):
        raise NotImplementedError


class FormantQuantiles(Analyzer):
    def GetName(self):
        return "FormantQuantiles"

    def RunAnalysis(self, df, group_name, output_dir):
        # output = df[['Filename']].copy()
        # output['Annotation'] = df[['Annotation']]
        df['barkF1_25p'] = df[['barkF1_3', 'barkF1_4']].mean(axis=1)
        df['barkF1_75p'] = df[['barkF1_8', 'barkF1_9']].mean(axis=1)
        df['barkF1_50p'] = df[['barkF1_6']]
        df['barkF2_25p'] = df[['barkF2_3', 'barkF2_4']].mean(axis=1)
        df['barkF2_75p'] = df[['barkF2_8', 'barkF2_9']].mean(axis=1)
        df['barkF2_50p'] = df[['barkF2_6']]
        df['diff_F1F1_25p'] = df['barkF1_25p'] - df['barkF2_25p']
        df['diff_F1F1_50p'] = df['barkF1_50p'] - df['barkF2_50p']
        df['diff_F1F1_75p'] = df['barkF1_75p'] - df['barkF2_75p']
        output_debug = pd.concat(
            [df[['Filename']],
             df[['Annotation']],
             df.loc[:, df.columns.str.startswith("barkF1")],
             df.loc[:, df.columns.str.startswith("barkF2")],
             df.loc[:, df.columns.str.startswith("diff")],
             ], axis=1)
        output = pd.DataFrame(
            df.loc[:, df.columns.str.startswith("diff")].mean()).T

        output_path = output_dir / (group_name + '.csv')
        output_debug_path = output_dir / (group_name + '.debug.csv')
        print(output_path)

        output_debug.to_csv(output_debug_path, index=False)
        output.to_csv(output_path, index=False)


class FormantRegression(Analyzer):
    def GetName(self):
        return "FormantRegression"

    def RunAnalysis(self, df, group_name, output_dir):
        s_f1 = df.loc[:, df.columns.str.startswith("barkF1")].mean()
        s_f2 = df.loc[:, df.columns.str.startswith("barkF2")].mean()
        # print(s_f1)
        x = np.arange(0, 9)
        y1 = s_f1['barkF1_2': 'barkF1_10'].to_numpy(dtype='float')
        y2 = s_f2['barkF2_2': 'barkF2_10'].to_numpy(dtype='float')
        coeff1 = np.polyfit(x, y1, 4)
        coeff2 = np.polyfit(x, y2, 4)
        line1 = np.poly1d(coeff1)
        line2 = np.poly1d(coeff2)
        line1d = np.polyder(line1, 1)
        line2d = np.polyder(line2, 1)
        line1dd = np.polyder(line1, 2)
        line2dd = np.polyder(line2, 2)
        line1dd_max = minimize_scalar(-line1dd, bounds=(0, 8), method='bounded')
        line2dd_max = minimize_scalar(-line2dd, bounds=(0, 8), method='bounded')
        inflection1 = line1dd_max.x
        inflection2 = line2dd_max.x
        df_inflex = pd.DataFrame(data={'f1_inflection': [inflection1], 'f2_inflection': [inflection2]})
        df_inflex.to_csv(output_dir / (group_name + '.csv') , index=False)

        plt.plot(x, y1, 'o')
        plt.plot(x, y2, 'x')
        plt.plot(x, line1(x), label='F1 fitted')
        plt.plot(x, line2(x), label='F2 fitted')
        plt.plot(x, line1d(x), label='F1 1st deriv')
        plt.plot(x, line2d(x), label='F2 1st deriv')
        plt.plot(x, line1dd(x), label='F1 2nd deriv')
        plt.plot(x, line2dd(x), label='F2 2nd deriv')
        plt.axvline(x=inflection1, linestyle=':', label='F1 inflection')
        plt.axvline(x=inflection2, linestyle='-.', label='F2 inflection')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.title(group_name)
        plt.savefig(output_dir / (group_name + '.png'), bbox_inches="tight")
        plt.clf()
        plt.cla()
        # print(output_dir)
        output_debug_path = output_dir / (group_name + '.debug.csv')
        df.to_csv(output_debug_path, index=False)
