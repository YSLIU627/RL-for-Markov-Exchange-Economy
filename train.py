from agent import Agent
from env import Env
import numpy as np
        
        
def train(BC = False,id=0,):
    np.random.seed(100+id)
    class Args:
            def __init__(self):
                self.lam = 1
                self.alpha = 0.5
                self.beta = 0.5  
                self.n = 3      
                self.hor = self.n**2# 9
                self.card_s = self.n**2 #9#20
                self.card_a = 5#3
                self.d_r = self.card_s*self.card_a
                self.n_demon = 5
                self.n_add = 1000
                self.train_num = 20
                self.expert_sim_time = 20#20#000
                self.kappa = 0.0001#1#0.001#5
                self.noise_factor = 0.1
                self.mu_max = 3.0
                self.mu_init_factor = 0
                self.verbose = False
                self.stepsize = 150
                
    
    args = Args()
    env = Env(args.hor,args.card_s,args.card_a,args.d_r,args.expert_sim_time,noise_factor = args.noise_factor,n=args.n,mu_init_factor= args.mu_init_factor)
    print("random performance",env._eval(None))
    agent = Agent(args.hor,args.card_s,args.card_a,args.d_r,env,args.n_demon,args.n_add,mu_max=args.mu_max,kappa =args.kappa,verbose=args.verbose,alpha=args.alpha,stepsize =args.stepsize)
    log = agent.train(args.train_num)
    
    if BC:
        from agent import BC_Agent
        print("BC on Expert Demonstration")
        agent2 = BC_Agent(args.hor,args.card_s,args.card_a,args.d_r,env,args.n_demon,args.n_add,mu_max=args.noise_factor)
        BC_rand = agent2.train(agent.expert_demon)
        agent3 = BC_Agent(args.hor,args.card_s,args.card_a,args.d_r,env,args.n_demon,args.n_add,mu_max=args.noise_factor)
        print("BC on Expert Demonstration + Additional Dataset")
        BC_exp = agent3.train(agent.add_data.mix(agent.expert_demon))
    
    exp_perform = env._eval(env.expert)
    rand_perform = env._eval(None)
    return [log,exp_perform, rand_perform, BC_exp, BC_rand]


def train_loop(num):
    print("start")
    keys = ["log","exp_perform", "rand_perform",  "BC_exp", "BC_rand"]
    data =  dict.fromkeys(keys)
    for key in keys:
        data[key] = []
    for id in range(num):
        temp_data = train(True,id)
        for i,key in enumerate(keys):
            data[key].append(temp_data[i])
    print("ends1")
    return data
train_loop(5)