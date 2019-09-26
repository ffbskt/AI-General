import numpy as np




class Node:
    parent = None  # parent Node
    value_sum = 0.  # sum of state values from all visits (numerator)
    times_visited = 0
    def __init__(self, parent=None, action=None, prob=None, env=None):
        self.parent = parent
        self.action = action
        self.P = prob
        self.children = set()
        if parent:
            self.immediate_reward, self.formula = env.do_move(parent.formula, action)
        else:
            self.immediate_reward, self.formula = 0.0, ''

    def is_leaf(self):
        return len(self.children) == 0

    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=10, max_value=1e5):  # 1e100
        """
        Computes ucb1 upper bound using current value and visit counts for node and it's parent.

        :param scale: Multiplies upper bound by that. From hoeffding inequality, assumes reward range to be [0,scale].
        :param max_value: a value that represents infinity (for unvisited nodes)

        """

        U = (self.P *
             np.sqrt(self.parent.times_visited) / (1 + self.times_visited))  # need if zero visited?

        # if self.get_mean_value() > 0:
        #    print(self.get_mean_value(), self.env.formula)

        # print(self.env.formula, U, self.P, self.parent.times_visited)
        return self.get_mean_value() + scale * U


class MCTS:
    def __init__(self, env, model, args):
        self.env = env
        self.model = model
        self.args = args
        node = Node()
        self.Nodes = {'': node}
        self.sum_reward = 0
        self.root = Node()


    def select_best_leaf(self):
        node = self.root
        while not node.is_leaf():
            node = max(node.children, key=lambda x: x.ucb_score())
        return node

    def expand(self, node, action_priors):
        for action, prob in enumerate(action_priors):
            node.children.add(Node(node, action, prob, self.env))

    def propagate(self, node):
        while node.parent:
            child_val = node.immediate_reward
            node = node.parent
            node.value_sum += child_val
            node.times_visited += 1

    def sampling(self):
        i, r = 0, 0
        while (self.args.iters > i and not r) or 2 * self.args.iters > i:
            node = self.select_best_leaf()
            r = node.immediate_reward
            #if node.parent:
            self.Nodes[node.formula] = node
            if r:
                self.propagate(node)
            else:
                pred_P, pred_R = self.model.predict(self.env.get_observation(node.formula))
                self.expand(node, pred_P)

            r = max(0, r)
            i += 1

        return  self.Nodes.values()

if __name__ == "__main__":
    import models
    from env_test import Env

    model = models.Model()
    env = Env(inp=1, out=2)

    #from pipeline import dotdict
    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    args = dotdict({'cpuct': 2, 'iters': 20})

    rsmp = MCTS(env, model, args)
    rand_val = list(rsmp.sampling())

    print([(f.formula, f.immediate_reward) for f in rand_val])

