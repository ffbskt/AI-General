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
        for l in formula[ind - 1:ind + window_size - 1]:
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

    def env_reset_new(self, formula=''):
        new_env = self.__class__(formula, self.inp, self.out)
        new_env.err = 0
        new_env.init_result()
        return new_env

    def do_move(self, action):
        self.formula += self.action_space[action]
        self.calc_formula()

        # print('env', self.formula, self.err, calc_formula(self.formula, self.inp)[1])

    def game_end(self):
        # return err, rew
        self.calc_formula()
        if self.err:
            return -1
        if self.result['o'] == self.out and not self.err:
            return 1
        is_end = 'o' in self.formula or len(self.formula) > 24 or self.err
        if is_end and self.result['o'] != self.out:
            return -1
        return 0

    def calc_formula(self):
        self.err = 0
        self.init_result()
        self.result['i'] = self.inp
        for ind, comand in enumerate(self.formula):
            try:

                if comand == 'e' and self.formula[ind + 1] in self.result:
                    self.result[self.formula[ind + 1]] = self.result[self.formula[ind - 1]]
                if comand == 's' and self.formula[ind + 1] in self.result:
                    self.result[self.formula[ind + 1]] += self.result[self.formula[ind - 1]]
                # if comand=='s':
                #    print('-', self.formula, end=' ')
            except IndexError:
                pass
            except (KeyError, TypeError, IndexError):
                self.err = 1

    def get_observation(self):
        self.calc_formula()
        net_observ = self.NN_input()
        return net_observ

    def one_hot_last(self, n_last=4):
        act_sp = self.action_space
        matrix = np.zeros([n_last * len(act_sp)])
        for i, f in enumerate(self.formula[-n_last:]):
            k = n_last - min(len(self.formula), n_last)  # start from last line
            matrix[act_sp.index(f) + (k + i) * 8] = 1
        return matrix

    def NN_input(self):
        # values, err = calc_formula(formula, inp, env.result)
        values_norm = np.array(list(self.result.values()))
        inp, out, err = self.inp, self.out, self.err
        if np.sum(values_norm) != 0:
            values_norm = values_norm / np.sum(values_norm)
        s = inp + out
        inp = inp / s
        out = out / s
        if values_norm.shape[0] != 6:
            print('ERROR', self.result, values_norm.shape, self.one_hot_last().shape)
        head = np.hstack([np.array([inp, out, err]), values_norm, self.one_hot_last()])
        return head


if __name__ == "__main__":
    fstr = 'ieo'
    fstr = 'ieo'  # '1eAAsAAsAisAAeo'
    # print(emb_formula(fstr, formula_dict))
    # print(emb_command(fstr, 3, command_dict).values())
    # print(Env.calc_formula())

    e = Env(fstr)
    e.do_move(6)
    print(e.game_end(), e.formula)
    print(e.get_observation())
    print(e.env_reset_new(), e.env_reset_new().result)