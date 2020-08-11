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


class Analyzer(object):
    def RunAnalysis(self, df, group_name, output_base_dir):
        raise NotImplementedError

    def GetName(self):
        raise NotImplementedError


class FormantQuantiles(Analyzer):
    def GetName(self):
        return "FormantQuantiles"

    def GetInputType(self):
        return "Formant"

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

        output_debug.to_csv(output_debug_path, index=False)
        output.to_csv(output_path, index=False)


class FormantQuantilesByDemographic(Analyzer):
    def GetName(self):
        return "FormantQuantilesByDemographic"

    def GetInputType(self):
        return "Formant"

    def RunAnalysis(self, df, outer_filters, inner_filters, group_name, output_dir):
        kBarWidth = 0.2
        for outer_f in outer_filters:
            key = outer_f.GetValue()
            matched_rows = dict()
            for _, row in df.iterrows():
                if not outer_f.IsMatched(row):
                    continue
                for inner_f in inner_filters:
                    if inner_f.IsMatched(row):
                        matched_rows.setdefault(
                            inner_f.GetValue(), []).append(row)
            if len(matched_rows) == 0:
                continue
            x = np.arange(3)
            for k, v in matched_rows.items():
                matched_df = pd.DataFrame(v)
                full_group_name = group_name + '@' + outer_f.GetValue() + '@@' + k
                df_mean = self.ComputeMean(
                    matched_df, full_group_name, output_dir)
                y = [df_mean['diff_F1F2_25p'][0],
                     df_mean['diff_F1F2_50p'][0],
                     df_mean['diff_F1F2_75p'][0]]
                plt.bar(x, y, width=kBarWidth, label=k)
                x = [xval + kBarWidth for xval in x]
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.xticks([r + kBarWidth for r in range(3)],
                       ('25%', '50%', '75%'))
            plt.title(key)
            plt.savefig(output_dir / (group_name + '@' +
                                      key + '.png'), bbox_inches="tight")
            plt.clf()
            plt.cla()

    def ComputeMean(self, df, full_group_name, output_dir):
        df['barkF1_25p'] = df[['barkF1_3', 'barkF1_4']].mean(axis=1)
        df['barkF1_75p'] = df[['barkF1_8', 'barkF1_9']].mean(axis=1)
        df['barkF1_50p'] = df[['barkF1_6']]
        df['barkF2_25p'] = df[['barkF2_3', 'barkF2_4']].mean(axis=1)
        df['barkF2_75p'] = df[['barkF2_8', 'barkF2_9']].mean(axis=1)
        df['barkF2_50p'] = df[['barkF2_6']]
        df['diff_F1F2_25p'] = df['barkF1_25p'] - df['barkF2_25p']
        df['diff_F1F2_50p'] = df['barkF1_50p'] - df['barkF2_50p']
        df['diff_F1F2_75p'] = df['barkF1_75p'] - df['barkF2_75p']

        output = pd.DataFrame(
            df.loc[:, df.columns.str.startswith("diff")].mean()).T

        output_path = output_dir / (full_group_name + '.csv')
        output_debug_path = output_dir / (full_group_name + '.debug.csv')

        output.to_csv(output_path, index=False)
        df.to_csv(output_debug_path, index=False)
        return output


class FormantRegression(Analyzer):
    def GetName(self):
        return "FormantRegression"

    def GetInputType(self):
        return "Formant"

    def RunAnalysis(self, df, group_name, output_dir):
        s_f1 = df.loc[:, df.columns.str.startswith("barkF1")].mean()
        s_f2 = df.loc[:, df.columns.str.startswith("barkF2")].mean()
        x = np.arange(0, 9)
        y1 = s_f1['barkF1_2': 'barkF1_10'].to_numpy(dtype='float')
        y2 = s_f2['barkF2_2': 'barkF2_10'].to_numpy(dtype='float')
        coeff1 = np.polyfit(x, y1, 4)
        coeff2 = np.polyfit(x, y2, 4)
        line1 = np.poly1d(coeff1)
        line2 = np.poly1d(coeff2)
        # line1d = np.polyder(line1, 1)
        # line2d = np.polyder(line2, 1)
        line1dd = np.polyder(line1, 2)
        line2dd = np.polyder(line2, 2)
        line1dd_max = minimize_scalar(-line1dd,
                                      bounds=(0, 8), method='bounded')
        line2dd_max = minimize_scalar(-line2dd,
                                      bounds=(0, 8), method='bounded')
        inflection1 = line1dd_max.x
        inflection2 = line2dd_max.x
        df_inflex = pd.DataFrame(
            data={'f1_inflection': [inflection1], 'f2_inflection': [inflection2]})
        df_inflex.to_csv(output_dir / (group_name + '.csv'), index=False)

        # Plot f1/f2
        plt.plot(x, y1, 'o')
        plt.plot(x, y2, 'x')
        plt.plot(x, line1(x), label='F1 fitted')
        plt.plot(x, line2(x), label='F2 fitted')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(group_name)
        plt.savefig(output_dir / (group_name + '.fitted.png'),
                    bbox_inches="tight")
        plt.clf()
        plt.cla()
        # plt.plot(x, line1d(x), label='F1 1st deriv')
        # plt.plot(x, line2d(x), label='F2 1st deriv')
        # Plot deriv and inflection
        plt.plot(x, line1dd(x), label='F1 2nd deriv')
        plt.plot(x, line2dd(x), label='F2 2nd deriv')
        plt.axvline(x=inflection1, linestyle=':', label='F1 inflection')
        plt.axvline(x=inflection2, linestyle='-.', label='F2 inflection')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(group_name)
        plt.savefig(output_dir / (group_name + '.inflection.png'),
                    bbox_inches="tight")
        plt.clf()
        plt.cla()
        output_debug_path = output_dir / (group_name + '.debug.csv')
        df.to_csv(output_debug_path, index=False)


class HnrRegression(Analyzer):
    def GetName(self):
        return "HnrRegression"

    def GetInputType(self):
        return "HNR"

    def RunAnalysis(self, df, group_name, output_dir):
        for i in range(1, 10):
            df['mid_'+str(i)] = df[['HNR_'+str(i),
                                    'HNR_'+str(i+1)]].mean(axis=1)
        sy = df.loc[:, df.columns.str.startswith('mid_')].mean()
        y = sy['mid_1': 'mid_9'].to_numpy(dtype='float')
        x = np.arange(0, 9)
        coeff = np.polyfit(x, y, 4)
        line1 = np.poly1d(coeff)
        line1dd = np.polyder(line1, 2)
        line1dd_max = minimize_scalar(-line1dd,
                                      bounds=(0, 8), method='bounded')
        inflection = line1dd_max.x
        df_inflex = pd.DataFrame(data={'inflection': [inflection]})
        df_inflex.to_csv(output_dir / (group_name + '.csv'), index=False)

        plt.plot(x, y, 'o')
        plt.plot(x, line1(x), label='fitted')
        plt.plot(x, line1dd(x), label='2nd deriv')
        plt.axvline(x=inflection, linestyle=':', label='inflection')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(group_name)
        plt.savefig(output_dir / (group_name + '.png'), bbox_inches="tight")
        plt.clf()
        plt.cla()

        output_debug_path = output_dir / (group_name + '.debug.csv')
        df.to_csv(output_debug_path, index=False)


class HnrQuantilesMean(Analyzer):
    def GetName(self):
        return "HnrQuantilesMean"

    def GetInputType(self):
        return "HNR"

    def RunAnalysis(self, df, group_name, output_dir):
        df['HNR_p25'] = df[['HNR_2', 'HNR_3']].mean(axis=1)
        df['HNR_p75'] = df[['HNR_7', 'HNR_8']].mean(axis=1)
        df['HNR_p50'] = df[['HNR_5']]
        output = pd.DataFrame(
            df.loc[:, df.columns.str.startswith("HNR_p")].mean()).T

        output_path = output_dir / (group_name + '.csv')
        output.to_csv(output_path, index=False)

        output_debug_path = output_dir / (group_name + '.debug.csv')
        df.to_csv(output_debug_path, index=False)


class HnrTTest(Analyzer):
    def GetName(self):
        return "HnrTTest"

    def GetInputType(self):
        return "HNR"

    def RunAnalysis(self, df, group_name, output_dir):
        df['HNR_25p'] = df[['HNR_2', 'HNR_3']].mean(axis=1)
        df['HNR_75p'] = df[['HNR_7', 'HNR_8']].mean(axis=1)
        df['HNR_50p'] = df[['HNR_5']]
        output = pd.DataFrame(
            df.loc[:, df.columns.str.startswith("diff")].mean()).T

        output_path = output_dir / (group_name + '.csv')
        output.to_csv(output_path, index=False)

        output_debug_path = output_dir / (group_name + '.debug.csv')
        df.to_csv(output_debug_path, index=False)
