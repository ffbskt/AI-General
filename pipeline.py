


for i in range(4):
        #m = MCTS_best_leaf(env, model, args)
        #val = list(m.sampling())
        #print(np.sum([m.Nodes[f].fin_reward > 0 for f in m.Nodes]), [(f.formula, f.predR) for f in get_batch(val)])
    rsmp = RandomSample(env, model, args)
    rand_val = list(rsmp.sampling())

    train_model(rand_val)