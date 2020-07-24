import filter as ft
import analyzer as az

GROUP_A = ([
    [ft.IsMale(), ft.IsFemale()],
    [ft.IsChild(), ft.IsYouth(), ft.IsAdult(), ft.IsSenior()],
    [ft.IsA1(), ft.IsA2()],
], [
    az.FormantQuantiles(),
    az.FormantRegression(),
])
