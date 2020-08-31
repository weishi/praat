import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize_scalar

import filter


kBarWidth = 0.2


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


def ComputeF1F2Diff(df):
    df['barkF1_25p'] = df[['barkF1_3', 'barkF1_4']].mean(axis=1)
    df['barkF1_75p'] = df[['barkF1_8', 'barkF1_9']].mean(axis=1)
    df['barkF2_25p'] = df[['barkF2_3', 'barkF2_4']].mean(axis=1)
    df['barkF2_75p'] = df[['barkF2_8', 'barkF2_9']].mean(axis=1)
    df['diff_F1_7525'] = df['barkF1_75p'] - df['barkF1_25p']
    df['diff_F2_7525'] = df['barkF2_75p'] - df['barkF2_25p']
    return df


class FormantQuantilesF1F2Base(Analyzer):
    def __init__(self, filter_map):
        self.filter_map = filter_map

    def RunAnalysis(self, df, group_name, output_dir):
        matched_rows_map = {}
        for key, _ in self.filter_map.items():
            matched_rows_map[key] = []

        for _, row in df.iterrows():
            for key, filters in self.filter_map.items():
                is_all_matched = [f.IsMatched(row) for f in filters]
                if np.all(is_all_matched):
                    matched_rows_map[key].append(row)
        matched_df = {}
        for key, rows in matched_rows_map.items():
            matched_df[key] = pd.DataFrame(rows)

        x = np.arange(2)
        for key, mdf in matched_df.items():
            mdf = ComputeF1F2Diff(mdf)
            df_mean = pd.DataFrame(
                mdf.loc[:, mdf.columns.str.startswith("diff")].mean()).T
            mdf.to_csv(output_dir / (group_name + '@@@' +
                                     key + '.debug.csv'), index=False)
            df_mean.to_csv(output_dir / (group_name + '@@@' +
                                         key+'Mean.debug.csv'), index=False)
            y = [df_mean['diff_F1_7525'][0], df_mean['diff_F2_7525'][0]]
            plt.bar(x, y, width=kBarWidth, label=key)
            x = [xval + kBarWidth for xval in x]

        plt.xticks([r + kBarWidth for r in range(2)], ('delta_F1', 'delta_F2'))
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(group_name)
        plt.savefig(output_dir / (group_name + '.png'), bbox_inches="tight")
        plt.clf()
        plt.cla()


class FormantQuantilesF1F2SaSb(FormantQuantilesF1F2Base):
    def __init__(self):
        super().__init__({
            'Sa': [filter.IsShanghainese(), filter.IsPosition('a')],
            'Sb': [filter.IsShanghainese(), filter.IsPosition('b')],
        })


class FormantQuantilesF1F2SbMb(FormantQuantilesF1F2Base):
    def __init__(self):
        super().__init__({
            'Sb': [filter.IsShanghainese(), filter.IsPosition('b')],
            'Mb': [filter.IsMandarin(), filter.IsPosition('b')],
        })


class FormantQuantilesSbMbBase(Analyzer):
    def __init__(self, formant):
        self.formant = formant

    def RunAnalysis(self, df, group_name, output_dir):
        rows_sb = []
        rows_mb = []
        for _, row in df.iterrows():
            if filter.IsShanghainese().IsMatched(row) and filter.IsPosition('b').IsMatched(row):
                rows_sb.append(row)
                continue
            if filter.IsMandarin().IsMatched(row) and filter.IsPosition('b').IsMatched(row):
                rows_mb.append(row)
                continue

        df_sb = pd.DataFrame(rows_sb)
        df_sb = ComputeF1F2Diff(df_sb)
        df_sb_avg = pd.DataFrame(
            df_sb.loc[:, df_sb.columns.str.startswith("diff")].mean()).T
        df_sb.to_csv(output_dir / (group_name +
                                   '@@@Sb.debug.csv'), index=False)
        df_sb_avg.to_csv(output_dir / (group_name +
                                       '@@@SbMean.debug.csv'), index=False)

        df_mb = pd.DataFrame(rows_mb)
        df_mb = ComputeF1F2Diff(df_mb)
        df_mb_avg = pd.DataFrame(
            df_mb.loc[:, df_mb.columns.str.startswith("diff")].mean()).T
        df_mb.to_csv(output_dir / (group_name +
                                   '@@@Mb.debug.csv'), index=False)
        df_mb_avg.to_csv(output_dir / (group_name +
                                       '@@@MbMean.debug.csv'), index=False)

        x = ['Sb', 'Mb']
        y = [df_sb_avg['diff_' + self.formant + '_7525'][0],
             df_mb_avg['diff_'+self.formant+'_7525'][0]]
        plt.bar(x, y, width=kBarWidth)

        plt.title(group_name)
        plt.savefig(output_dir / (group_name + '.png'), bbox_inches="tight")
        plt.clf()
        plt.cla()


class FormantQuantilesF1SbMb(FormantQuantilesSbMbBase):
    def __init__(self):
        super().__init__('F1')


class FormantQuantilesF2SbMb(FormantQuantilesSbMbBase):
    def __init__(self):
        super().__init__('F2')


class FormantRegressionBase(Analyzer):
    def __init__(self, filters):
        self.filters = filters

    def RunAnalysis(self, df, group_name, output_dir):
        matched_rows = []
        for _, row in df.iterrows():
            is_all_matched = [f.IsMatched(row) for f in self.filters]
            if np.all(is_all_matched):
                matched_rows.append(row)
        df = pd.DataFrame(matched_rows)

        filter_name = '_'.join([f.GetValue() for f in self.filters])
        full_group_name = group_name + '@@' + filter_name
        s_f1 = df.loc[:, df.columns.str.startswith("barkF1")].mean()
        s_f2 = df.loc[:, df.columns.str.startswith("barkF2")].mean()
        x = np.arange(0, 9)
        y1 = s_f1['barkF1_2': 'barkF1_10'].to_numpy(dtype='float')
        y2 = s_f2['barkF2_2': 'barkF2_10'].to_numpy(dtype='float')
        coeff1 = np.polyfit(x, y1, 4)
        coeff2 = np.polyfit(x, y2, 4)
        line1 = np.poly1d(coeff1)
        line2 = np.poly1d(coeff2)
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
        df_inflex.to_csv(output_dir / (full_group_name + '.csv'), index=False)

        # Plot f1/f2
        plt.plot(x, y1, 'o')
        plt.plot(x, y2, 'x')
        plt.plot(x, line1(x), label='F1 fitted')
        plt.plot(x, line2(x), label='F2 fitted')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(full_group_name)
        plt.savefig(output_dir / (full_group_name + '.fitted.png'),
                    bbox_inches="tight")
        plt.clf()
        plt.cla()
        # Plot deriv and inflection
        plt.plot(x, line1dd(x), label='F1 2nd deriv')
        plt.plot(x, line2dd(x), label='F2 2nd deriv')
        plt.axvline(x=inflection1, linestyle=':', label='F1 inflection')
        plt.axvline(x=inflection2, linestyle='-.', label='F2 inflection')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(full_group_name)
        plt.savefig(output_dir / (full_group_name + '.inflection.png'),
                    bbox_inches="tight")
        plt.clf()
        plt.cla()
        output_debug_path = output_dir / (full_group_name + '.debug.csv')
        df.to_csv(output_debug_path, index=False)


class FormantRegressionSa(FormantRegressionBase):
    def __init__(self):
        super().__init__([filter.IsShanghainese(), filter.IsPosition('a')])


class FormantRegressionSb(FormantRegressionBase):
    def __init__(self):
        super().__init__([filter.IsShanghainese(), filter.IsPosition('b')])


class FormantRegressionMb(FormantRegressionBase):
    def __init__(self):
        super().__init__([filter.IsMandarin(), filter.IsPosition('b')])


class FormantInflectionBase(Analyzer):
    def __init__(self, filter_map):
        self.filter_map = filter_map

    def RunAnalysis(self, df, group_name, output_dir):
        matched_rows_map = {}
        for key, _ in self.filter_map.items():
            matched_rows_map[key] = []

        for _, row in df.iterrows():
            for key, filters in self.filter_map.items():
                is_all_matched = [f.IsMatched(row) for f in filters]
                if np.all(is_all_matched):
                    matched_rows_map[key].append(row)
        matched_df = {}
        for key, rows in matched_rows_map.items():
            matched_df[key] = pd.DataFrame(rows)

        x_all = []
        f1_front = []
        f1_back = []
        f2_front = []
        f2_back = []
        for key, mdf in matched_df.items():
            s_f1 = mdf.loc[:, mdf.columns.str.startswith("barkF1")].mean()
            s_f2 = mdf.loc[:, mdf.columns.str.startswith("barkF2")].mean()
            x = np.arange(0, 9)
            y1 = s_f1['barkF1_2': 'barkF1_10'].to_numpy(dtype='float')
            y2 = s_f2['barkF2_2': 'barkF2_10'].to_numpy(dtype='float')
            coeff1 = np.polyfit(x, y1, 4)
            coeff2 = np.polyfit(x, y2, 4)
            line1 = np.poly1d(coeff1)
            line2 = np.poly1d(coeff2)
            line1dd = np.polyder(line1, 2)
            line2dd = np.polyder(line2, 2)
            line1dd_max = minimize_scalar(-line1dd,
                                          bounds=(0, 8), method='bounded')
            line2dd_max = minimize_scalar(-line2dd,
                                          bounds=(0, 8), method='bounded')
            inflection1 = line1dd_max.x
            inflection2 = line2dd_max.x
            x_all.append(key)
            f1_front.append(inflection1 / 8.0)
            f1_back.append(1 - inflection1 / 8.0)
            f2_front.append(inflection2 / 8.0)
            f2_back.append(1 - inflection2 / 8.0)

        full_group_name = group_name + '@@' + '_'.join(matched_df.keys())
        plt.bar(x_all, f1_front, width=kBarWidth, label='Front')
        plt.bar(x_all, f1_back, bottom=np.array(
            f1_front), width=kBarWidth, label='Back')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.title(full_group_name+'@@@F1')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig(output_dir / (full_group_name + '.f1.png'), bbox_inches="tight")
        plt.clf()
        plt.cla()

        plt.bar(x_all, f2_front, width=kBarWidth, label='Front')
        plt.bar(x_all, f2_back, bottom=np.array(
            f2_front), width=kBarWidth, label='Back')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.title(full_group_name+'@@@F2')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig(output_dir / (full_group_name + '.f2.png'), bbox_inches="tight")
        plt.clf()
        plt.cla()


class FormantInflectionMb(FormantInflectionBase):
    def __init__(self):
        super().__init__({'Mb': [filter.IsMandarin(), filter.IsPosition('b')]})


class FormantInflectionSbMb(FormantInflectionBase):
    def __init__(self):
        super().__init__({
            'Sb': [filter.IsShanghainese(), filter.IsPosition('b')],
            'Mb': [filter.IsMandarin(), filter.IsPosition('b')]
        })

class FormantInflectionSaSb(FormantInflectionBase):
    def __init__(self):
        super().__init__({
            'Sa': [filter.IsShanghainese(), filter.IsPosition('a')],
            'Sb': [filter.IsShanghainese(), filter.IsPosition('b')],
        })

def GetAge(row):
    comps = row['Filename'].split('_')
    assert len(comps) == 5 or len(comps) == 6
    age_gender = int(comps[2])
    if 1 <= age_gender <= 20:
        return '1.Senior'
    if 21 <= age_gender <= 40:
        return '2.Adult'
    if 41 <= age_gender <= 60:
        return '3.Youth'
    if 61 <= age_gender <= 80:
        return '4.Child'
    raise NotImplementedError


def GetGender(row):
    comps = row['Filename'].split('_')
    assert len(comps) == 5 or len(comps) == 6
    age_gender = int(comps[2])
    if age_gender % 2 == 0:
        return '1.Female'
    else:
        return '2.Male'


class FormantQuantilesSlicedBase(Analyzer):
    def __init__(self, formant, word, word_filters, slicer):
        self.formant = formant
        self.word = word
        self.word_filters = word_filters
        self.slicer = slicer

    def RunAnalysis(self, df, group_name, output_dir):
        matched_rows_map = {}
        for _, row in df.iterrows():
            is_all_matched = [f.IsMatched(row) for f in self.word_filters]
            if np.all(is_all_matched):
                matched_rows_map.setdefault(self.slicer(row), []).append(row)

        x = []
        y = []
        full_group_name = group_name + '@@' + self.formant+'_'+self.word
        for key, matched_rows in matched_rows_map.items():
            mdf = pd.DataFrame(matched_rows)
            mdf = ComputeF1F2Diff(mdf)
            df_mean = pd.DataFrame(
                mdf.loc[:, mdf.columns.str.startswith("diff")].mean()).T
            mdf.to_csv(output_dir / (full_group_name + '@@@' +
                                     key + '.debug.csv'), index=False)
            df_mean.to_csv(output_dir / (full_group_name + '@@@' +
                                         key+'Mean.debug.csv'), index=False)
            x.append(key)
            y.append(df_mean['diff_'+self.formant+'_7525'][0])

        plt.bar(x, y)
        plt.title(full_group_name)
        plt.savefig(output_dir / (full_group_name + '.png'),
                    bbox_inches="tight")
        plt.clf()
        plt.cla()

class FormantQuantilesF1SaAge(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Sa', [filter.IsShanghainese(), filter.IsPosition('a')],
                         GetAge)


class FormantQuantilesF1SbAge(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Sb', [filter.IsShanghainese(), filter.IsPosition('b')],
                         GetAge)


class FormantQuantilesF2SaAge(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Sa', [filter.IsShanghainese(), filter.IsPosition('a')],
                         GetAge)


class FormantQuantilesF2SbAge(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Sb', [filter.IsShanghainese(), filter.IsPosition('b')],
                         GetAge)


class FormantQuantilesF1MbAge(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Mb', [filter.IsMandarin(), filter.IsPosition('b')],
                         GetAge)


class FormantQuantilesF2MbAge(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Mb', [filter.IsMandarin(), filter.IsPosition('b')],
                         GetAge)

class FormantQuantilesF1SaGender(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Sa', [filter.IsShanghainese(), filter.IsPosition('a')],
                         GetGender)

class FormantQuantilesF1SbGender(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Sb', [filter.IsShanghainese(), filter.IsPosition('b')],
                         GetGender)

class FormantQuantilesF2SaGender(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Sa', [filter.IsShanghainese(), filter.IsPosition('a')],
                         GetGender)

class FormantQuantilesF2SbGender(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Sb', [filter.IsShanghainese(), filter.IsPosition('b')],
                         GetGender)


class FormantQuantilesF1MbGender(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Mb', [filter.IsMandarin(), filter.IsPosition('b')],
                         GetGender)


class FormantQuantilesF2MbGender(FormantQuantilesSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Mb', [filter.IsMandarin(), filter.IsPosition('b')],
                         GetGender)

class FormantRegressionSlicedBase(Analyzer):
    def __init__(self, word, word_filters, slicer):
        self.word = word
        self.word_filters = word_filters
        self.slicer = slicer

    def RunAnalysis(self, df, group_name, output_dir):
        matched_rows_map = {}
        for _, row in df.iterrows():
            is_all_matched = [f.IsMatched(row) for f in self.word_filters]
            if np.all(is_all_matched):
                matched_rows_map.setdefault(self.slicer(row), []).append(row)

        full_group_name = group_name + '@@' + self.word
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z_label = sorted(matched_rows_map.keys())
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(z_label)))
        for key, matched_rows in matched_rows_map.items():
            mdf = pd.DataFrame(matched_rows)
            s_f1 = mdf.loc[:, mdf.columns.str.startswith("barkF1")].mean()
            s_f2 = mdf.loc[:, mdf.columns.str.startswith("barkF2")].mean()
            x = np.arange(0, 9)
            y1 = s_f1['barkF1_2': 'barkF1_10'].to_numpy(dtype='float')
            y2 = s_f2['barkF2_2': 'barkF2_10'].to_numpy(dtype='float')
            coeff1 = np.polyfit(x, y1, 4)
            coeff2 = np.polyfit(x, y2, 4)
            line1 = np.poly1d(coeff1)
            line2 = np.poly1d(coeff2)
            line1dd = np.polyder(line1, 2)
            line2dd = np.polyder(line2, 2)
            line1dd_max = minimize_scalar(-line1dd,
                                          bounds=(0, 8), method='bounded')
            line2dd_max = minimize_scalar(-line2dd,
                                          bounds=(0, 8), method='bounded')
            inflection1 = line1dd_max.x
            inflection2 = line2dd_max.x
            inflection1y = line1(inflection1)
            inflection2y = line2(inflection2)
            color = colors[z_label.index(key)]
            z = z_label.index(key)
            ax.plot(x, y1, zs=z, zdir='x', c=color, label='F1', linewidth=3.0)
            ax.plot(x, y2, zs=z, zdir='x', c=color, label='F2')
            ax.plot([inflection1, inflection1], [inflection1y-1, inflection1y+1], zs=z, zdir='x', c='black')
            ax.plot([inflection2, inflection2], [inflection2y-1, inflection2y+1], zs=z, zdir='x', c='black')

        ax.set(xticks=range(len(z_label)), xticklabels=z_label)
        plt.title(full_group_name)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig(output_dir / (full_group_name + '.png'),
                    bbox_inches="tight")
        plt.clf()
        plt.cla()

class FormantRegressionSaAge(FormantRegressionSlicedBase):
    def __init__(self):
        super().__init__('Sa', [filter.IsShanghainese(), filter.IsPosition('a')], GetAge)

class FormantRegressionSbAge(FormantRegressionSlicedBase):
    def __init__(self):
        super().__init__('Sb', [filter.IsShanghainese(), filter.IsPosition('b')], GetAge)

class FormantRegressionMbAge(FormantRegressionSlicedBase):
    def __init__(self):
        super().__init__('Mb', [filter.IsMandarin(), filter.IsPosition('b')], GetAge)

class FormantRegressionSaGender(FormantRegressionSlicedBase):
    def __init__(self):
        super().__init__('Sa', [filter.IsShanghainese(), filter.IsPosition('a')], GetGender)

class FormantRegressionSbGender(FormantRegressionSlicedBase):
    def __init__(self):
        super().__init__('Sb', [filter.IsShanghainese(), filter.IsPosition('b')], GetGender)

class FormantRegressionMbGender(FormantRegressionSlicedBase):
    def __init__(self):
        super().__init__('Mb', [filter.IsMandarin(), filter.IsPosition('b')], GetGender)



class FormantInflectionSlicedBase(Analyzer):
    def __init__(self, formant, word, word_filters, slicer):
        self.formant = formant
        self.word = word
        self.word_filters = word_filters
        self.slicer = slicer

    def RunAnalysis(self, df, group_name, output_dir):
        matched_rows_map = {}
        for _, row in df.iterrows():
            is_all_matched = [f.IsMatched(row) for f in self.word_filters]
            if np.all(is_all_matched):
                matched_rows_map.setdefault(self.slicer(row), []).append(row)

        x_all = []
        y_front = []
        y_back = []
        full_group_name = group_name + '@@' + self.formant+'_'+self.word
        for key, matched_rows in matched_rows_map.items():
            mdf = pd.DataFrame(matched_rows)
            formant_prefix = 'bark' + self.formant
            f = mdf.loc[:, mdf.columns.str.startswith(formant_prefix)].mean()
            x = np.arange(0, 9)
            y = f[formant_prefix + '_2': formant_prefix + '_10'].to_numpy(dtype='float')
            coeff = np.polyfit(x, y, 4)
            line = np.poly1d(coeff)
            linedd = np.polyder(line, 2)
            linedd_max = minimize_scalar(-linedd,
                                          bounds=(0, 8), method='bounded')
            inflection = linedd_max.x
            x_all.append(key)
            y_front.append(inflection / 8.0)
            y_back.append(1 - inflection / 8.0)

        full_group_name = group_name + '@@' + self.formant + '_' + self.word
        plt.bar(x_all, y_front, width=kBarWidth, label='Front')
        plt.bar(x_all, y_back, bottom=np.array(y_front), width=kBarWidth, label='Back')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(full_group_name)
        plt.savefig(output_dir / (full_group_name + '.png'),
                    bbox_inches="tight")
        plt.clf()
        plt.cla()

class FormantInflectionF1SbAge(FormantInflectionSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Sb', [filter.IsShanghainese(), filter.IsPosition('b')],
                         GetAge)

class FormantInflectionF2SbAge(FormantInflectionSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Sb', [filter.IsShanghainese(), filter.IsPosition('b')],
                         GetAge)

class FormantInflectionF1MbAge(FormantInflectionSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Mb', [filter.IsMandarin(), filter.IsPosition('b')],
                         GetAge)

class FormantInflectionF2MbAge(FormantInflectionSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Mb', [filter.IsMandarin(), filter.IsPosition('b')],
                         GetAge)

class FormantInflectionF1SbGender(FormantInflectionSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Sb', [filter.IsShanghainese(), filter.IsPosition('b')],
                         GetGender)

class FormantInflectionF2SbGender(FormantInflectionSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Sb', [filter.IsShanghainese(), filter.IsPosition('b')],
                         GetGender)

class FormantInflectionF1MbGender(FormantInflectionSlicedBase):
    def __init__(self):
        super().__init__('F1', 'Mb', [filter.IsMandarin(), filter.IsPosition('b')],
                         GetGender)

class FormantInflectionF2MbGender(FormantInflectionSlicedBase):
    def __init__(self):
        super().__init__('F2', 'Mb', [filter.IsMandarin(), filter.IsPosition('b')],
                         GetGender)