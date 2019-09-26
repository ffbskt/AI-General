import numpy as np
from env_test import Env
from models import Model
from collections import defaultdict
import heapq

def softmax(x):
    ex = np.exp(x)
    return ex / (np.sum(ex) or 1e-10)

class Node:
    def __init__(self, formula, predP, predR, fin_reward=0, Q=0, fin_prob=0):
        self.formula = formula
        self.predP = predP
        self.predR = predR
        self.fin_reward = fin_reward
        self.fin_prob = fin_prob
        self.N = 0
        self.Q = Q

    def __lt__(self, other):
        if self.fin_reward:
            return False
        if other.fin_reward:
            return True
        #return self.predR > other.predR # ??? predict R or computed Q
        return self.Q > other.Q

class Sampling_operator:
    """
    Interface.
    Put state and action prob to enviroment and get reward
    then save result of actions in current state in Nodes{state: information}
    then generate new state and action use model output and own algorithms
    """

    def __init__(self, env, model, args):
        self.env = env
        self.model = model
        self.args = args
        node = Node(formula='', predP=1, predR=0.5)
        self.Nodes = {'': node}
        self.sum_reward = 0

    def find_new_state(self):
        """

        :return: initial formula
        """
        pass

    def expand(self, state):
        """

        :return: add new states and information to Nodes
        """
        pass

    def sampling(self):
        """

        :return: if need recompute all sampling
        """
        pass


class RandomSampling(Sampling_operator):
    def __init__(self, env, model, args):
        Sampling_operator.__init__(self, env, model, args)

    def sampling(self):
        if self.Nodes[''].fin_prob == 0:
            del self.Nodes['']

        i, r = 0, 0
        while (self.args.iters > i and not r) or 2 * self.args.iters > i:
            formula_len = np.random.choice(range(24))
            formula = ''
            for step in range(formula_len):
                fin_prob = np.zeros(8)
                fin_reward = np.zeros(8)
                for a in range(8):
                    r, next_formula = self.env.do_move(formula, a)
                    fin_reward[a] = r
                    if r > 0:
                        fin_prob[a] = 1
                        pred_P, pred_R = self.model.predict(self.env.get_observation(formula))
                        if formula not in self.Nodes:
                            self.Nodes[formula] = Node(formula, pred_P, pred_R, fin_reward=fin_reward,
                                                       fin_prob=fin_prob)
                        break
                self.sum_reward += r
                if r > 0: break

                # choose action to move for next iteration
                pred_P, pred_R = self.model.predict(self.env.get_observation(formula))
                #if max(pred_R) >= 0: print(pred_R, pred_P)
                a = np.random.choice(range(8), p=pred_P)
                if max(pred_R) <= 0:
                    a = np.random.choice(range(8))

                fin_prob[a] = 1
                # save node with data
                if formula not in self.Nodes:
                    self.Nodes[formula] = Node(formula, pred_P, pred_R, fin_reward=fin_reward, fin_prob=fin_prob)

                #go to next_state
                r, formula = self.env.do_move(formula, a)
                # print(pred_P)

            r = max(0, r)
            i += 1

            #if r > 0:
            #    print('!!', formula)
            #self.recompute_r_p(0.01)


        return self.Nodes.values()

    def recompute_r_p(self, coef=1):
        for f in self.Nodes:
            if np.max(self.Nodes[f].fin_reward) > 0:

                for i in range(len(f) - 1):
                    if f[i+1] in self.Nodes:
                        ind = self.env.action_space.index(f[i+1])
                        self.Nodes[f[:i]].fin_reward[ind] += coef / (len(f) - i)
                        self.Nodes[f[:i]].fin_prob[ind] += coef / (len(f) - i)
                        self.Nodes[f[:i]].fin_prob = softmax(self.Nodes[f[:i]].fin_prob)





if __name__ == "__main__":
    model = Model()
    env = Env(inp=1, out=2)

    from pipeline import dotdict
    args = dotdict({'cpuct':2, 'iters':200})


    rsmp = RandomSampling(env, model, args)
    rand_val = list(rsmp.sampling())

    print(rand_val)