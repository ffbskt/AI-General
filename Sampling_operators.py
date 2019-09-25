import numpy as np
from env_test import Env
from policy import Model
from collections import defaultdict
import heapq

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class Node:
    def __init__(self, formula, predP, predR, fin_reward=0, Q=0):
        self.formula = formula
        self.predP = predP
        self.predR = predR
        self.fin_reward = fin_reward
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
        node = Node('', 1, 0.5)
        self.Nodes = {'': node}

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


