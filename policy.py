from env_test import Env #, calc_formula
import numpy as np
import torch
from torch import Tensor, nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


env = Env()
def softmax(x):
    #ex = np.exp(x)
    #return ex / (np.sum(ex) or 1e-10)
    return F.softmax(Variable(Tensor(x))).data.numpy()

def norm(x):
    #print(x)#, x.values())
    s = np.sum(x)
    #print(x)
    if s > 0:
        return x / s
    return x


class Policy:
    def __init__(self, env=env):
        self.env = env
        self.model = Model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.1, weight_decay=0.1)
        self.loss_backet = []
        
    def prepare_input(self, inp, out, formula):
        self.env.formula = formula
        self.env.inp, self.env.out = inp, out
        self.env.calc_formula()
        values, err = self.env.result, self.env.err 
        net_observ = NN_input(inp, out, formula, values, err)
        return net_observ

    def policy_value_function(self, inp, out, formula):
        # TODO net learning only on replay buffer...?? 
        # why not imediatly...
        net_observ = self.prepare_input(inp, out, formula)
        act_prob, val = self.model(Variable(Tensor(net_observ)))
        return act_prob.data.numpy(), val.data.numpy()

    def train_model(self, replay_buf):
        for i in range(200):
            batch = replay_buf.get_batch(batch_size=10)
            #print(batch)
            X = np.vstack([self.prepare_input(*sample[:3]) 
                              for sample in batch]) 
            value_sum = np.vstack([norm(list(sample[3].env.result.values())) for sample in batch])
            ucb_score = np.vstack([np.array([sample[4]]) for sample in batch])
            self.optimizer.zero_grad()
            
            
            p_pred, v_pred = self.model(Variable(Tensor(X)))
            #print(ucb_score.shape, torch.log(p_pred))
            #print(Variable(Tensor(value_sum)), v_pred)
            #we
            #if not i % 10:
                #print('loss:', v_pred, value_sum, torch.mean(Variable(Tensor(value_sum)) - v_pred) ** 2)
                #print('lprob:', p_pred, ucb_score)
            val_loss = torch.min(torch.mean(Variable(Tensor(value_sum)) - v_pred ** 2), Variable(Tensor([10])))
            loss = val_loss - torch.mean(Variable(Tensor(ucb_score)) * torch.log(p_pred))
            #loss = torch.mean(- Variable(Tensor(ucb_score)) * torch.log(p_pred))
            
            #print(loss, np.sum([torch.sum(p.data) for p in self.model.parameters()]))
            loss.backward()
            self.optimizer.step()
            self.loss_backet.append(loss.data.numpy())


#p = Policy()

#print(p.policy_value_function('formula'))


def one_hot_last(formula, n_last=4):
    act_sp = env.action_space
    matrix = np.zeros([n_last * len(act_sp)])
    for i, f in enumerate(formula[-n_last:]):
        matrix[act_sp.index(f) + i * 8] = 1
    return matrix



def NN_input(inp, out, formula, values, err):
    #values, err = calc_formula(formula, inp, env.result)
    values_norm = np.array(list(values.values()))
    if np.sum(values_norm) != 0:
        values_norm = values_norm / np.sum(values_norm)
    s = inp + out
    inp = inp / s
    out = out / s
    if values_norm.shape[0] != 6:
        print('ERROR', values, values_norm.shape, one_hot_last(formula).shape)
    head = np.hstack([np.array([inp, out, err]), values_norm, one_hot_last(formula)])
    return head

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(41, 82)
        self.fc2 = nn.Linear(82, 40)
        self.action_prob_out = nn.Linear(40, 8)
        #self.val0 = nn.Linear(40, 80)
        self.val = nn.Linear(40, 8)

        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.1, weight_decay=0.1)
        self.loss_backet = []

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        act_prob = F.softmax(self.action_prob_out(x), dim=0)
        val = F.tanh(self.val(x))
        #val_sum = self.val_sum_out(val)

        return act_prob, val

    def predict(self, x):
        x = Variable(Tensor(x))
        act_prob, val = self.forward(x)
        return act_prob.data.numpy(), val.data.numpy()

    def training(self, X, reward, probability):
        self.optimizer.zero_grad()
        p_pred, v_pred = self.model(Variable(Tensor(X)))
        val_loss = torch.min(torch.mean(Variable(Tensor(reward)) - v_pred ** 2), Variable(Tensor([10])))
        loss = val_loss - torch.mean(Variable(Tensor(probability)) * torch.log(p_pred))
        loss.backward()
        self.optimizer.step()
        self.loss_backet.append(loss.data.numpy())

   #def regularize(self):
   #    for w in model.parameters():
           
      

class ReplayBuffer:
    # TODO add pickle.dump()
    # TODO which value predict? how it change?
    """
       add node and save formula from parent then by parent data will predicted node.env.result and node.P
    """
    def __init__(self):
        self.replay = [] # (formula, value_sum, ucb_score)
        
    def get_states(self, node):
        for ch in node.children:
            self.add(ch)
            self.get_states(ch)

    def add(self, node): # which value we predict ??
        inp, out, = node.env.inp, node.env.out
        if node.parent:
            parent = node.parent
            formula = parent.env.formula
            ucb = softmax([ch.ucb_score() for ch in parent.children])
        
        
            self.replay.append((inp, out, formula, node, ucb))
    
    def get_batch(self, batch_size=10):
        n = len(self.replay)
        index = [
            np.random.randint(0, n - 1)
            for _ in range(batch_size)
        ]
        #print(len(self.replay), index, [self.replay[i] for i in index]   )
        return [self.replay[i] for i in index]    

    
        

if __name__ == "__main__":
    one_hot_last('1e1', 4).shape
    NN_input(1,1, 'Aio').shape

    p = Policy()
    print(p.policy_value_function(1,1,'Aso'))
    
    

