class Env:
    dictionary = ('1', 'i', 'o', 'A', 'B', 'C')
    operation = ('s', 'e')
    action_space = dictionary + operation
    n_actions = len(action_space)

    def __init__(self, inp=1, out=1):
        self.inp, self.out = inp, out
        self.result = {}
        self.err = 0

    def init_result(self):
        for val in self.dictionary:
            self.result[val] = 0

    def do_move(self, formula, action):
        formula += self.action_space[action]
        return self.game_end(formula), formula

    def game_end(self, formula):
        self.calc_formula(formula)
        if self.err:
            return -1
        if self.result['o'] == self.out and not self.err:
            return 1
        is_end = 'o' in formula or len(formula) > 24 or self.err
        if is_end and self.result['o'] != self.out:
            return -1
        return 0

    def calc_formula(self, formula):
        self.err = 0
        self.init_result()
        self.result['i'] = self.inp
        for ind, comand in enumerate(formula):
            try:
                if ind < 1: continue
                if comand == 'e' and formula[ind + 1] in self.result:
                    self.result[formula[ind + 1]] = self.result[formula[ind - 1]]
                if comand == 's' and formula[ind + 1] in self.result:
                    self.result[formula[ind + 1]] += self.result[formula[ind - 1]]
            except IndexError:
                pass
            except (KeyError, TypeError, IndexError):
                self.err = 1

    def get_observation(self, formula, time=0):
        result = self.game_end(formula)
        return result, self.result


if __name__ == "__main__":
    fstr = 'eBCAieo'
    e = Env(1, 1)
    result = e.do_move(fstr, 6)
    e.calc_formula(fstr)
    print(e.get_observation(fstr))
    print(e.get_observation('eBCA'))
