import numpy as np
import torch
from torch import Tensor, nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnv1 = nn.Conv1d(1,6,8)
        self.fc1 = nn.Linear(17, 182)
        self.fc2 = nn.Linear(182, 40)
        self.action_prob_out = nn.Linear(40, 8)
        #self.val0 = nn.Linear(40, 80)
        self.val = nn.Linear(40, 8)

        #self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.1, weight_decay=0.1)
        #self.loss_backet = []

    def forward(self, x):
        #print(x.shape)
        #x = x.view(-1,17)
        #x = self.cnv1(x.view(1, 1,-1))
        # print(x.shape)
        #x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        act_prob = F.softmax(self.action_prob_out(x), dim=-1)
        val = F.tanh(self.val(x))
        #val_sum = self.val_sum_out(val)

        return act_prob, val

    def predict(self, x):
        self.eval()
        x = Variable(Tensor(x))
        act_prob, val = self.forward(x)
        return act_prob.data.numpy(), val.data.numpy()




class Trainer:
    def __init__(self, env, batch_size=10):
        self.env = env
        self.batch_size = batch_size
        self.loss_backet = []

    def get_batch(self, nodes_buc, batch_size=0):
        n = len(nodes_buc)
        index = [
            np.random.randint(0, n - 1)
            for _ in range(batch_size or self.batch_size)
        ]
        #print(len(self.replay), index, [self.replay[i] for i in index]   )
        return [nodes_buc[i] for i in index]

    def transform_bach_as_input(self, batch):
        X = []
        real_prob = []
        real_reward = []
        for node in batch:
            self.env.calc_formula(node.formula)
            net_observ = self.env.NN_input(node.formula)
            X.append(net_observ)
            real_prob.append(node.fin_prob) # matrix size(8, 1) of next (f+a) prob
            real_reward.append(node.fin_reward)
        X = np.vstack(X)
        real_prob = np.vstack(real_prob)
        real_reward = np.vstack(real_reward)
        return X, real_reward, real_prob




    def train_model(self, nodes_buc, model, batch_size=0, net_iters=200):
        self.optimizer = optim.SGD(model.parameters(), lr=0.9, momentum=0.3, weight_decay=0.1)

        for i in range(net_iters):
            batch = self.get_batch(nodes_buc, batch_size=batch_size or self.batch_size)
            # print(batch)

            X, real_reward, real_prob = self.transform_bach_as_input(batch)

            self.optimizer.zero_grad()
            #print(i, X.shape)
            model.train()
            p_pred, v_pred = model(Variable(Tensor(X)))
            # print('pr  ', probability, 'pp  ', p_pred)
            val_loss = torch.mean((Variable(Tensor(real_reward)) - v_pred) ** 2)  # , Variable(Tensor([10]))
            #loss = val_loss - torch.mean(Variable(Tensor(real_prob)) * torch.log(p_pred))
            loss = - torch.mean(Variable(Tensor(real_prob)) * torch.log(p_pred))
            loss.backward()
            self.optimizer.step()
            self.loss_backet.append(loss.data.numpy())


class mctsTrainer(Trainer):
    def __init__(self, env, mcts, batch_size=10):
        Trainer.__init__(self, env=env, batch_size=batch_size)
        self.mcts = mcts

    def transform_bach_as_input(self, batch):
        X = []
        real_prob = []
        real_reward = []
        for node in batch:
            self.env.calc_formula(node.formula)
            net_observ = self.env.NN_input(node.formula)
            X.append(net_observ)
            prob = np.zeros(self.env.n_actions)
            reward = np.zeros(self.env.n_actions)
            for i, a in enumerate(self.env.action_space):
                if node.formula + a in self.mcts.Nodes:
                    prob[i] = self.mcts.Nodes[node.formula + a].fin_prob
                    g = self.mcts.Nodes[node.formula + a].ucb_score()
                    # print(g, node.formula + a, node.formula, a, self.mcts.Nodes)
                    reward[i] = g
                #print()
            real_prob.append(prob) # matrix size(8, 1) of next (f+a) prob
            real_reward.append(reward)
        X = np.vstack(X)
        real_prob = np.vstack(real_prob)
        real_reward = np.vstack(real_reward)
        return X, real_reward, real_prob


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.aprob_pred = nn.Linear(hidden_dim, vocab_size)
        self.val_pred = nn.Linear(hidden_dim, vocab_size)
        self.result_pred = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        slen, b_sz, ind_s = sentence.shape
        embeds = self.word_embeddings(sentence)
        # print(embeds.shape, embeds.view(len(sentence), 1, -1))
        lstm_out, hidden = self.lstm(embeds.view(len(sentence), 1, self.embedding_dim))
        hidden = hidden[0].view(b_sz, self.hidden_dim)
        aprob_pred = F.log_softmax(self.aprob_pred(hidden), dim=1)
        val_pred = self.val_pred(hidden)
        result_pred = self.result_pred(hidden)
        return aprob_pred, val_pred, result_pred

    def predict(self, x):
        self.eval()
        # x = Variable(torch.LongTensor(x))
        act_prob, val, _ = self.forward(x)
        return act_prob.data.numpy(), val.data.numpy()

    def get_observation(self, formula, env):
        return Variable(torch.LongTensor(np.array([env.action_space.index(a) for a in formula])))


#model = LSTMModel(4, 32, 8, 8)

if __name__ == "__main__":
    from env_test import Env
    from mcts import MCTS

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    model = Model()
    model = LSTMModel(4, 32, 8, 8)
    env = Env()

    args = dotdict({'cpuct':0.5, 'iters':1000})
    rsmp = MCTS(env, model, args)
    val = list(rsmp.sampling())
    print(len(val))
    val = [v for v in val if v.parent and v.parent.times_visited > 2]
    print(len(val))
    t = mctsTrainer(env, rsmp, batch_size=50)
    t.train_model(val, model, net_iters=300)
    #examples = deque([], maxlen=1000)

    batch = t.get_batch(val, batch_size=5)
    for b in t.transform_bach_as_input(batch):
        print(b)
    import matplotlib.pyplot as plt

    plt.plot(t.loss_backet)
    plt.show()