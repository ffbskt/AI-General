import numpy as np
from env_test import Env
from policy import Model
import heapq

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class Node:
    def __init__(self, formula, predP, predR, fin_reward=0):
        self.formula = formula
        self.predP = predP
        self.predR = predR
        self.fin_reward = fin_reward
        self.N = 0
        self.Q = 0

class MCTS_best_leaf:
    """
        different algorithm. The Idea do not go down by tree choose best ucb node, except - we immediatly
        compare only leafs and go to best leaf even parents branch has low ucb.

    """
    def __init__(self, env, model, args):
        self.env = env
        node = Node(env.formula, 1, 0.5)
        self.heap_best = heapq.heapify([(0, ),])
        self.Ns = {node.formula: 0}
        self.model = model

    def find_best_leaf(self):
        return self.heap_best.heappop()

    def expand(self, node):
        pred_P, pred_R = self.model.predict(self.env.get_observation(node.formula))
        self.up_prev_N(node.formula)
        for a in range(self.env.n_actions):
            r, next_formula = self.env.step(node.formula, a)
            Q = self.compute_Q()

    def up_prev_N(self, formula):
        for i in len(formula):
            self.Ns[formula[:-i]] += 1

    def compute_Q(self, pred_P, pred_R, formula):



























































if __name__ == "__main__":
    policy_net = Policy()
    for i in range(1):
        root = Node(None, None, 1.)
        #rew = plan_mcts(root, policy_net, replay_buffer, n_iters=500)
        #policy_net.train_model(replay_buffer)
        #reward.extend(rew)

    import matplotlib.pyplot as plt
    #print(policy_net.loss_backet)    
    plt.plot(policy_net.loss_backet)
    plt.show()
    


# while root.children:
    #print(root.children.pop().children.pop().env.formula)
#    root = root.select_best_leaf()
#    print('ff', root.env.formula)
