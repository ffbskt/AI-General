import numpy as np
import torch
from torch import Tensor, nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from models import Trainer


class mctsTrainer(Trainer):
    def __init__(self, env, mcts, batch_size=80):
        Trainer.__init__(self, env=env, batch_size=batch_size)
        self.mcts = mcts

    def get_sorted_batch(self, val, batch_size=0):
        sort_val = sorted(val, key=lambda x: len(x.formula))
        batch_size = batch_size or self.batch_size
        i = np.random.randint(len(val) - batch_size)
        return sort_val[i:i + batch_size]

    def get_batch(self, nodes_buc, batch_size=0):
        return self.get_sorted_batch(nodes_buc, batch_size or self.batch_size)

    # def transform_bach_as_input(self, batch):
    #     X = []
    #     real_prob = []
    #     real_reward = []
    #     for node in batch:
    #
    #
    #         X.append([])
    #         #XX = np.zeros(self.env.n_actions, )
    #         prob = np.zeros(self.env.n_actions)
    #         reward = np.zeros(self.env.n_actions)
    #         for i, a in enumerate(self.env.action_space):
    #             if node.formula + a in self.mcts.Nodes:
    #                 self.env.calc_formula(node.formula + a)
    #                 net_observ = self.env.NN_input(node.formula + a)
    #                 X[-1].append(net_observ)
    #                 prob[i] = self.mcts.Nodes[node.formula + a].fin_prob
    #                 g = self.mcts.Nodes[node.formula + a].ucb_score()
    #                 # print(g, node.formula + a, node.formula, a, self.mcts.Nodes)
    #                 reward[i] = g
    #             #print()
    #         real_prob.append(prob) # matrix size(8, 1) of next (f+a) prob
    #         real_reward.append(reward)
    #     #X = np.vstack(X)
    #     real_prob = np.vstack(real_prob)
    #     real_reward = np.vstack(real_reward)
    #     return X, real_reward, real_prob

    def train_model(self, nodes_buc, model, batch_size=0, net_iters=200):
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.3, weight_decay=0.1)

        for i in range(net_iters):
            batch = self.get_batch(nodes_buc, batch_size=batch_size or self.batch_size)
            # print(batch)

            X, real_reward, real_prob, cresult = self.transform_bach_as_input(batch, model)
            X = X.view(-1, self.batch_size)


            #print(i, real_prob.shape)
            model.train()
            #for x, rr, rp in zip(X, real_reward, real_prob):
                #print(xx, rrr,rpp)
            self.optimizer.zero_grad()
            #print(123, X.shape)#model(Variable(Tensor(X))))
            p_pred, v_pred, result_pred = model(X)
            # print('pr  ', probability, 'pp  ', p_pred)
            #print(cresult)
            val_loss = torch.mean((Variable(Tensor(real_reward)) - v_pred) ** 2)  # , Variable(Tensor([10]))
            result_loss = torch.mean(torch.sum((Variable(Tensor(cresult))/8 - result_pred/8) ** 2))
            loss = val_loss #- torch.mean(Variable(Tensor(real_prob)) * torch.log(p_pred))# + result_loss 
            #loss = - torch.mean(Variable(Tensor(real_prob)) * torch.log(p_pred))
            #print(loss, rpp, p_pred)
            loss.backward()
            self.optimizer.step()
            self.loss_backet.append(loss.data.numpy())


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMModel, self).__init__()
        self.MAXL = 25
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size + 1

        self.word_embeddings = nn.Embedding(self.vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.aprob_pred = nn.Linear(hidden_dim, self.vocab_size-1)
        self.val_pred = nn.Linear(hidden_dim, 1)
        self.result_pred = nn.Linear(hidden_dim, self.vocab_size - 3)

    def forward(self, sentence):
        slen, b_sz = sentence.shape
        #print(b_sz)
        embeds = self.word_embeddings(sentence.view(slen, b_sz))
        #print(embeds.shape, embeds.view(len(sentence), 1, -1))
        lstm_out, hidden = self.lstm(embeds)#.view(len(sentence), 1, self.embedding_dim))
        hidden = hidden[0].view(b_sz, self.hidden_dim)
        aprob_pred = F.softmax(self.aprob_pred(hidden), dim=1)
        val_pred = self.val_pred(hidden)
        result_pred = self.result_pred(hidden)
        #print(234, embeds.shape, sentence.shape, hidden.shape, b_sz)
        return aprob_pred, val_pred, result_pred

    def predict(self, x):
        self.eval()
        # x = Variable(torch.LongTensor(x))
        act_prob, val, _ = self.forward(x)
        return act_prob.data.numpy(), val.data.numpy()

    def get_observation(self, formula, env, time=0):
        #print(formula)
        seq = np.zeros(self.MAXL).reshape(-1, 1) + 8
        if not formula:
            formula_index = np.array([8]).reshape([-1, 1])
        else:
            formula_index = np.array([env.action_space.index(a) for a in formula])
        seq[:formula_index.shape[0], 0] = formula_index
        #print(seq.shape)
        return Variable(torch.LongTensor(seq))


if __name__ == "__main__":
    from env_test import Env
    from mcts import MCTS

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    model = LSTMModel(4, 32, 8, 8)
    env = Env()

    args = dotdict({'cpuct':0.5, 'iters':1000})
    mcts = MCTS(env, model, args)
    val = list(mcts.sampling())
    print(len(val))
    val = [v for v in val if v.parent and v.parent.times_visited > 2]
    print(len(val))


    t = mctsTrainer(env, mcts, batch_size=10)

    #t = Trainer(env, batch_size=20)
    for i in range(3):
        print(i, t.loss_backet[-3:])
        mcts = MCTS(env, model, args)
        val = list(mcts.sampling())
        print(len(val), val[-1].history_data['time'], val[-1].formula) 
        val = t.clean_unpredict_node(val)
        #print('aa', val)
        t.train_model(val, model, net_iters=300)
    # #examples = deque([], maxlen=1000)
    #
    #batch = t.get_batch(val)
    #print(list(t.transform_bach_as_input(batch)[1]))

    import matplotlib.pyplot as plt
    #
    plt.plot(t.loss_backet)
    plt.show()
