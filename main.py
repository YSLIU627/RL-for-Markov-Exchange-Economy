from agent import Agent
from env import env
import numpy as np


class Args:
    def __init__(self):
        self.n = 3     
        self.card_s = 10
        self.card_a = 5
        self.N = 3
        self.L = 3
        self.hor = 3

        self.kappa = 0.005 # ucb coefficient
        self.delta = 0.1 # probability 1-delta
        self.alpha = 0.00001 # thompson sampling coefficient
        
        self.freq = 1
        self.T = 5
        self.train_num = 200
        self.num_sample = 500


args = Args()


def train(args, id):
    np.random.seed(311+id)
    environment = env(args.n, args.card_s, args.card_a, args.N, args.L, args.hor)
    agent = Agent(args.card_s, args.card_a, args.N, args.L, args.hor, args.T, args.delta, args.alpha, args.kappa, environment, args.freq, args.num_sample)

    return agent.train_policy(args.train_num, 0.5)


def train_loop(num):

    print("======Start======")
    
    keys = ["nu_pi_perform", "nu_random_perform", "nu_piopt_perform", "nuopt_pioptnuopt_perform", "R_planner", "R_random", "R_total"]
    data =  dict.fromkeys(keys)

    for key in keys:
        data[key] = []
    for id in range(num):
        print("======Train id:", id, '======')
        temp_data = train(args, id)
        for i, key in enumerate(keys):
            data[key].append(temp_data[i])

    print("======End======")
    return data

