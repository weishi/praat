import itertools

class BaseFilter(object):
    def GetValue(self):
        raise NotImplementedError

    def IsMatched(self, row):
        raise NotImplementedError

    def GetAgeGender(self, row):
        comps = row['Filename'].split('_')
        assert len(comps) == 5 or len(comps) == 6
        age_gender = int(comps[2])
        return age_gender


class IsMale(BaseFilter):
    def GetValue(self):
        return "Male"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return age_gender % 2 == 1


class IsFemale(BaseFilter):
    def GetValue(self):
        return "Female"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return age_gender % 2 == 0


class IsSenior(BaseFilter):
    def GetValue(self):
        return "Senior1_20"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return 1 <= age_gender <= 20


class IsAdult(BaseFilter):
    def GetValue(self):
        return "Adult21_40"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return 21 <= age_gender <= 40


class IsYouth(BaseFilter):
    def GetValue(self):
        return "Youth41_60"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return 41 <= age_gender <= 60


class IsChild(BaseFilter):
    def GetValue(self):
        return "Child61_80"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return 61 <= age_gender <= 80


class IsA1(BaseFilter):
    def GetValue(self):
        return "a1"

    def IsMatched(self, row):
        annotation = row['Annotation']
        return annotation == 'a1'


class IsA2(BaseFilter):
    def GetValue(self):
        return "a2"

    def IsMatched(self, row):
        annotation = row['Annotation']
        return annotation == 'a2'


# group_filters = itertools.product(*GROUP_A)
# for gf in group_filters:
#   name = [f.GetValue() for f in gf]
#   print('@'.join(name))

# row = dict()
# row['Filename'] = "M_a_05_01_a"
# row['Annotation'] = "a1"
