import itertools


class BaseFilter(object):
    def GetValue(self):
        raise NotImplementedError

    def GetType(self):
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

    def GetType(self):
        return "Gender"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return age_gender % 2 == 1


class IsFemale(BaseFilter):
    def GetValue(self):
        return "Female"

    def GetType(self):
        return "Gender"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return age_gender % 2 == 0


class IsSenior(BaseFilter):
    def GetValue(self):
        return "Senior1_20"

    def GetType(self):
        return "Age"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return 1 <= age_gender <= 20


class IsAdult(BaseFilter):
    def GetValue(self):
        return "Adult21_40"

    def GetType(self):
        return "Age"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return 21 <= age_gender <= 40


class IsYouth(BaseFilter):
    def GetValue(self):
        return "Youth41_60"

    def GetType(self):
        return "Age"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return 41 <= age_gender <= 60


class IsChild(BaseFilter):
    def GetValue(self):
        return "Child61_80"

    def GetType(self):
        return "Age"

    def IsMatched(self, row):
        age_gender = BaseFilter.GetAgeGender(self, row)
        return 61 <= age_gender <= 80


class IsVariant(BaseFilter):
    def __init__(self, variant):
        self.variant = variant

    def GetValue(self):
        return self.variant

    def GetType(self):
        return "Variant"

    def IsMatched(self, row):
        annotation = row['Annotation']
        return annotation == self.variant


class IsWordNum(BaseFilter):
    def __init__(self, word_num):
        assert len(word_num) > 0
        self.word_num = word_num

    def GetValue(self):
        return 'Word' + '_'.join(str(x) for x in self.word_num)

    def GetType(self):
        return "WordNum"

    def IsMatched(self, row):
        comps = row['Filename'].split('_')
        assert len(comps) == 5 or len(comps) == 6
        row_word_num = int(comps[3])
        return row_word_num in self.word_num


class IsPosition(BaseFilter):
    def __init__(self, pos):
        self.pos = pos

    def GetValue(self):
        assert self.pos in ['a', 'b']
        if self.pos == 'a':
            return 'Front'
        else:
            return 'Back'

    def GetType(self):
        return "Position"

    def IsMatched(self, row):
        comps = row['Filename'].split('_')
        assert len(comps) == 5 or len(comps) == 6
        row_pos = comps[4]
        assert row_pos in ['a', 'b']
        return row_pos == self.pos

class IsShanghainese(BaseFilter):
    def GetValue(self):
        return "Shanghainese"

    def GetType(self):
        return "Accent"

    def IsMatched(self, row):
        return row['Filename'].startswith('S')

class IsMandarin(BaseFilter):
    def GetValue(self):
        return "Mandarin"

    def GetType(self):
        return "Accent"

    def IsMatched(self, row):
        return row['Filename'].startswith('M')