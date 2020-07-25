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


class IsVariant(BaseFilter):
    def __init__(self, variant):
        self.variant = variant

    def GetValue(self):
        return self.variant

    def IsMatched(self, row):
        annotation = row['Annotation']
        return annotation == self.variant

class IsWordNum(BaseFilter):
    def __init__(self, word_num):
        self.word_num = word_num

    def GetValue(self):
        return 'WordNum' + str(self.word_num)

    def IsMatched(self, row):
        comps = row['Filename'].split('_')
        assert len(comps) == 5 or len(comps) == 6
        row_word_num = int(comps[3])
        return row_word_num == self.word_num


class IsPosition(BaseFilter):
    def __init__(self, pos):
        self.pos = pos

    def GetValue(self):
        assert self.pos in ['a', 'b']
        if self.pos == 'a':
            return 'Front'
        else:
            return 'Back'

    def IsMatched(self, row):
        comps = row['Filename'].split('_')
        assert len(comps) == 5 or len(comps) == 6
        row_pos = int(comps[4])
        return row_pos == self.pos
