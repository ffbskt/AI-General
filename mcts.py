import numpy as np
from gym.core import Wrapper
from pickle import dumps,loads
from collections import namedtuple
from policy import Policy, ReplayBuffer
from env_test import Env

#a container for get_result function below. Works just like tuple, but prettier
ActionResult = namedtuple("action_result",("snapshot","observation","reward","is_done","info"))



    

    
class Node:
    """ a tree node for MCTS """
    
    #metadata:
    parent = None          #parent Node
    value_sum = 0.         #sum of state values from all visits (numerator)
    times_visited = 0      #counter of visits (denominator)

    
    def __init__(self,parent,action, prob):
        """
        Creates and empty node with no children.
        Does so by commiting an action and recording outcome.
        
        :param parent: parent Node
        :param action: action to commit from parent Node
        
        """
        
        self.parent = parent
        self.action = action
        self.P = prob
        self.val_pred = 0
        self.children = set()       #set of child nodes


        if parent:
            self.env = Env(parent.env.formula, inp=1, out=2)
            self.env.do_move(action)
            #print('1', self.env.result)
            self.is_done, self.immediate_reward = self.env.game_end()
            #self.immediate_reward *= 1000
            #if self.immediate_reward:
            #    print('2', self.env.formula, self.is_done, self.env.game_end(), self.env.result)
        else:
            self.env = Env()
            self.is_done, self.immediate_reward = 0, 0



        #get action outcome and save it

        ##res = env.do_move(parent.state, action)
        
        ##self.state,self.observation,self.immediate_reward,self.is_done,_ = res
        #if self.is_done:
        #    self.value_sum = self.immediate_reward
        #    self.times_visited = 1
        #    print('ucb', self.ucb_score(), self.get_mean_value(), self.value_sum, self.immediate_reward)
        
        #if self.immediate_reward > 0:
        #    print("Node", self.immediate_reward, self.is_done, self.ucb_score())
            
        
        
        
    def is_leaf(self):
        return len(self.children)==0
    
    def is_root(self):
        return self.parent is None
    
    def get_mean_value(self): 
        return self.value_sum / self.times_visited if self.times_visited !=0 else 0
    
    def ucb_score(self,scale=10, max_value=1e5): # 1e100
        """
        Computes ucb1 upper bound using current value and visit counts for node and it's parent.
        
        :param scale: Multiplies upper bound by that. From hoeffding inequality, assumes reward range to be [0,scale].
        :param max_value: a value that represents infinity (for unvisited nodes)
        
        """
        
        #if self.times_visited == 0: ???
        #    return max_value
        
        #compute ucb-1 additive component (to be added to mean value)
        #hint: you can use self.parent.times_visited for N times node was considered,
        # and self.times_visited for n times it was visited
        
        ##U = np.sqrt(2 * np.log(self.parent.times_visited) / self.times_visited)
        #if self.get_mean_value() > 0:
        #    print(self.get_mean_value(), self.action)

        U = (self.P *
                   np.sqrt(self.parent.times_visited) / (1 + self.times_visited)) # need if zero visited?

        #if self.get_mean_value() > 0:
        #    print(self.get_mean_value(), self.env.formula)
        
        #print(self.env.formula, U, self.P, self.parent.times_visited)
        return self.get_mean_value() + scale*U
    
    
    #MCTS steps
    
    def select_best_leaf(self):
        """
        Picks the leaf with highest priority to expand
        Does so by recursively picking nodes with best UCB-1 score until it reaches the leaf.
        
        """
        if self.is_leaf():
            return self
        
        children = self.children
        
        # add random?? or desision
        best_child = max(children, key=lambda x: x.ucb_score())
        
        return best_child.select_best_leaf()
    
    def expand(self, action_priors):
        """
        Expands the current node by creating all possible child nodes.
        Then returns one of those children.
        """
        
        assert not self.is_done, "can't expand from terminal state"
        #if action:
            
        #    self.children.add(Node(self,action, env))
        #    return self.select_best_leaf()

        #probability = policy_value_function(self.env.formula)

        for action, prob in enumerate(action_priors):
            #
            self.children.add(Node(self, action, prob))
        
        # return self.select_best_leaf()
    
    
    def propagate(self, child_value):
        """
        Uses child value (sum of rewards) to update parents recursively.
        """
        #compute node value
        my_value = self.immediate_reward + child_value ## TODO dicrease imediate_reward each step
        
        #update value_sum and times_visited 
        # Not sum in determinate env
        #self.value_sum += my_value
        #if my_value > 0:
        #    print("get_reward! p", my_value, not self.is_root())
        # TODO write justheuristic about bug
        ## self.value_sum = max(my_value, self.value_sum)
        self.value_sum += my_value #(self.value_sum - my_value) / self.times_visited
        self.times_visited+=1
        
        #propagate upwards
        if not self.is_root():
            self.parent.propagate(my_value)
        
    def safe_delete(self):
        """safe delete to prevent memory leak in some python versions"""
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child
            
    
    
def plan_mcts(root, policy_net, replay_buffer, n_iters=1000, t_max=10):
    """
    builds tree with monte-carlo tree search for n_iters iterations
    :param root: tree node to plan from
    :param n_iters: how many select-expand-simulate-propagete loops to make
    """
    reward = [0,]

    for _ in range(n_iters):
        #print([(int(i.ucb_score()), i.env.formula, i.times_visited) for i in root.children], )
        
        node = root.select_best_leaf()
        #if node.times_visited > 100:
        #    print('best leaf', node.times_visited, node.env.formula, node.P, node.ucb_score())
        #    print([(i.ucb_score(), i.env.formula)  for i in node.parent.children])
        
        replay_buffer.add(node)
        reward.append(reward[-1] + node.immediate_reward)  
        if node.is_done:
            if node.immediate_reward > 0:
                print("get_reward mcts plan!", node.env.formula)
            
            node.propagate(0)

            
            #env.reset()
       
        else: #node is not terminal
            action_prob, val = policy_net.policy_value_function(node.env.inp, node.env.out, node.env.formula)
            
            ## if _ < 30:
            ##    action_prob = np.random.dirichlet((1,1,1,1,1,1,1,1), 1)[0]
            #print(env.formula, node.P, action_prob)
            #action_prob = np.random.randn(8)
            node.expand(action_prob)
        
    return reward

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class MCTS:
    def __init__(self, root, model, args):
        self.root = root
        self.model = model
        self.args = args
        self.Examples = []

    #def sampling(self):
    #   for i in self.args.numMCTSSims:


if __name__ == "__main__":
    policy_net = Policy()
    replay_buffer = ReplayBuffer()
    reward = []
    for i in range(1):
        root = Node(None, None, 1.)
        rew = plan_mcts(root, policy_net, replay_buffer, n_iters=500)
        policy_net.train_model(replay_buffer)
        reward.extend(rew)

    import matplotlib.pyplot as plt
    #print(policy_net.loss_backet)    
    plt.plot(policy_net.loss_backet)
    plt.plot(reward)
    plt.show()
    


# while root.children:
    #print(root.children.pop().children.pop().env.formula)
#    root = root.select_best_leaf()
#    print('ff', root.env.formula)
