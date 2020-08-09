import filter as ft
import analyzer as az

GROUP_A = ([
    [ft.IsShanghainese(), ft.IsMandarin()],
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('a1'), ft.IsVariant('a2')],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
])

GROUP_C = ([
    [ft.IsShanghainese(), ft.IsMandarin()],
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('c1'),
     ft.IsVariant('c2'),
     ft.IsVariant('c2vs'),
     ft.IsVariant('c2h'),
     ft.IsVariant('c4')],
    [ft.IsWordNum([1, 2]),
     ft.IsWordNum([3, 5, 6]),
     ft.IsWordNum([7, 8, 9]),
     ft.IsWordNum([10]),
     ft.IsWordNum([11, 12, 13, 14, 15])],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
    az.HnrRegression(),
    # az.HnrQuantilesMean(),
])

GROUP_D1 = ([
    [ft.IsShanghainese(), ft.IsMandarin()],
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('d1'), ft.IsVariant('d2')],
    [ft.IsWordNum([1, 3]),
     ft.IsWordNum([4]),
     ],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
    az.HnrRegression(),
    # az.HnrQuantilesMean()
])

GROUP_D2 = ([
    [ft.IsShanghainese(), ft.IsMandarin()],
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('d2n'), ft.IsVariant('d2h')],
    [ft.IsWordNum([5, 6, 7, 11, 12, 13, 14]), ],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
    az.HnrRegression(),
    # az.HnrQuantilesMean()
])
