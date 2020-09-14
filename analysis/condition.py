import numpy as np

class Condition(object):
    def __init__(self, user_condition_arr, word_condition_arr=[]):
        self.user_condition = []
        for uc in user_condition_arr:
          conds = uc.split(',')
          if len(self.user_condition) == 0:
            for i in range(len(conds)):
              self.user_condition.append(set())
          for i, c in enumerate(conds):
            self.user_condition[i].add(c)
        self.word_condition = set()
        for wc in word_condition_arr:
          self.word_condition.add(wc)
        word_name = 'all'
        if len(self.word_condition) > 0:
          word_name = 'w'.join(self.word_condition)
        self.group_name = '@'.join(user_condition_arr) + '@' + word_name
        

    def IsMatchedUser(self, key, vals):
        is_any_matched = []
        for uc_set in self.user_condition:
          is_matched = True
          for uc in uc_set:
            if uc not in vals:
              is_matched = False
          is_any_matched.append(is_matched)
        if not np.any(is_any_matched):
          return False
        if len(self.word_condition) > 0:
          if self.GetWord(key) not in self.word_condition:
            return False
        return True

    def GetGroupName(self):
      return self.group_name

    # key: a_01_03
    # value: S_a_b_a
    def GetWord(self, key):
        comps = key.split('_')
        assert len(comps) == 3
        return comps[2]

    def GetPosition(self, row):
        comps = row['Filename'].split('_')
        row_pos = comps[4]
        assert row_pos in ['a', 'b']
        return row_pos

    def GetLanguage(self, row):
        comps = row['Filename'].split('_')
        return comps[0]

    # def IsMatchedRow(self, row):
    #   if self.row_condition == 'SaSb':
    #     if self.GetLanguage(row) == 'S':
    #       return (True, 'S' + self.GetPosition(row))
    #     return (False, '')
    #   if self.row_condition == 'SbMb':
    #     if self.GetPosition(row) == 'b':
    #       return (True, self.GetLanguage(row) + 'b')
    #     return (False, '')
    #   raise NotImplementedError

