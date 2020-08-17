class Condition(object):
    def __init__(self, user_condition_arr, row_condition):
        self.user_condition = set()
        for uc in user_condition_arr:
          self.user_condition.add(uc)
        assert row_condition == 'SaSb' or row_condition == 'SbMb'
        self.row_condition = row_condition
        self.group_name = '@'.join(user_condition_arr) + '@@' + row_condition
        

    def IsMatchedUser(self, vals):
        for c in self.user_condition:
            if c not in vals:
                return False
        return True

    def GetGroupName(self):
      return self.group_name

    def GetPosition(self, row):
        comps = row['Filename'].split('_')
        row_pos = comps[4]
        assert row_pos in ['a', 'b']
        return row_pos

    def GetLanguage(self, row):
        comps = row['Filename'].split('_')
        return comps[0]

    def IsMatchedRow(self, row):
      if self.row_condition == 'SaSb':
        if self.GetLanguage(row) == 'S':
          return (True, 'S' + self.GetPosition(row))
        return (False, '')
      if self.row_condition == 'SbMb':
        if self.GetPosition(row) == 'b':
          return (True, self.GetLanguage(row) + 'b')
        return (False, '')
      raise NotImplementedError

CONDITION_A = [
  Condition(['S_a_a_a1', 'S_a_b_a1'], 'SaSb'), 
  Condition(['S_a_a_a1', 'S_a_b_a2'], 'SaSb'),
  Condition(['S_a_a_a1', 'S_a_b_a2'], 'SbMb'),
]