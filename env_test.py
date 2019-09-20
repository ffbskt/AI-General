from collections import defaultdict
import numpy as np

values = {'1': 1, 'i': 0, 'o': 0, 'A': 0, 'B': 0, 'C': 0}
formula_dict = defaultdict(lambda: defaultdict(int))
command_dict = defaultdict(lambda: defaultdict(int))


def init_dict(dd, keys=list(values) + ['s', 'e']):
    for v in keys:
        for v2 in keys:
            formula_dict[v][v2]
            command_dict[v][v2]

init_dict(formula_dict)
init_dict(command_dict)




def emb_formula(formula, dd, val):

    for command in val:
        for cur_command in formula:
            dd[command][cur_command] += 1
    return dd

def emb_command(formula, window_size, dd):
    for ind, c in enumerate(formula):
        for l in formula[ind-1:ind + window_size-1]:
            dd[c][l] += 1
    return dd

class Env:
    dictionary = ('1', 'i', 'o', 'A', 'B', 'C')
    operation = ('s', 'e')
    action_space = dictionary + operation
    n_actions = len(action_space)

    def __init__(self, formula='', inp=1, out=1):
        self.formula = formula
        self.inp, self.out = inp, out
        self.result = {}
        
        # self.obs = np.zeros([2 * self.n_actions + 1, self.n_actions])
        self.err = 0

    def init_result(self):
        for val in self.dictionary:
            self.result[val] = 0


    def do_move(self, action):
        self.formula += self.action_space[action]
        self.calc_formula()

        # print('env', self.formula, self.err, calc_formula(self.formula, self.inp)[1])

    def game_end(self): 
        # return err, rew
        self.calc_formula()
        if self.err:
            return 1, -1
        if self.result['o'] == self.out and not self.err:
            return 1, 10
        is_end = 'o' in self.formula or len(self.formula) > 24 or self.err
        if is_end and self.result['o'] != self.out:
            return 1, -0.8
        return 0, 0

    def calc_formula(self):
        self.err = 0
        self.init_result()
        self.result['i'] = self.inp
        for ind, comand in enumerate(self.formula):
            try:
                if comand == 'e' and self.formula[ind + 1] in self.result:
                    self.result[self.formula[ind + 1]]=self.result[self.formula[ind - 1]]
                if comand == 's' and self.formula[ind + 1] in self.result:
                    self.result[self.formula[ind + 1]] += self.result[self.formula[ind - 1]]
            except IndexError:
                pass
            except (KeyError, TypeError, IndexError):
                self.err = 1
        




        
if __name__ == "__main__":
    fstr = 'ieo'
    fstr = 'iCAo'  #'1eAAsAAsAisAAeo'
    #print(emb_formula(fstr, formula_dict))
    #print(emb_command(fstr, 3, command_dict).values())
    #print(Env.calc_formula())

    e = Env(fstr)
    e.do_move(6)
    print(e.game_end())
