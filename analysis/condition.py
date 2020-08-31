class Condition(object):
    def __init__(self, user_condition_arr, word_condition_arr=[]):
        self.user_condition = set()
        for uc in user_condition_arr:
          self.user_condition.add(uc)
        self.word_condition = set()
        for wc in word_condition_arr:
          self.word_condition.add(wc)
        # assert row_condition == 'SaSb' or row_condition == 'SbMb'
        # self.row_condition = row_condition
        word_name = 'all'
        if len(self.word_condition) > 0:
          word_name = 'w'.join(self.word_condition)
        self.group_name = '@'.join(user_condition_arr) + '@' + word_name
        

    def IsMatchedUser(self, key, vals):
        for c in self.user_condition:
            if c not in vals:
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

