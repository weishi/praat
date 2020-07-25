import filter as ft
import analyzer as az

GROUP_A = ([
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('a1'), ft.IsVariant('a2')],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
])

GROUP_C1 = ([
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('c1')],
    [ft.IsWordNum(i) for i in range(1, 15)],
    [ft.IsPosition('a'), ft.IsPosition('b')],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
    az.HnrRegression(),
])

GROUP_C2 = ([
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('c2'), ft.IsVariant('c4')],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
])

GROUP_D1 = ([
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('d1')],
    [ft.IsWordNum(i) for i in range(1, 15)],
    [ft.IsPosition('a'), ft.IsPosition('b')],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
])

GROUP_D2 = ([
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsVariant('c2')],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
])
