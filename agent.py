import numpy as np
from ce import utilityFunc, agentPolicy, agentPolicySample


class Agent():
    def __init__(self, card_s, card_a, N, L, hor, T, delta, alpha, kappa, env, freq, num_sample):
        self.card_s = card_s
        self.card_a = card_a
        self.N = N
        self.L = L
        self.hor = hor
        self.T = T #number of PRD iterations
        self.delta = delta #probability
        self.alpha = alpha #optimistic coefficient for utiltiy estimation
        self.kappa = kappa #optimistic coefficient for future social welfare

        self.env = env
        self.freq = freq

        self.tot = np.zeros((self.hor, self.card_s, self.card_a))
        self.vis = np.zeros((self.hor, self.card_s, self.card_a, self.card_s))
        self.x_buffer = [] #shape: (card_s, hor, length_of_buffer, N, L)
        self.u_buffer = [] #shape: (card_s, hor, length_of_buffer, N)
        self._Lambda = np.zeros((self.card_s, self.N, self.hor, self.L, self.L))
        self._R = np.zeros((self.card_s, self.N, self.hor, self.L))
        self._A_total_bar = np.zeros((self.N, self.hor, self.card_s, self.L))
        self._A_total_opt = np.zeros((self.N, self.hor, self.card_s, self.L))

        self.P_estimation_0 = np.ones_like(self.vis)
        for h in range(self.hor):
            for s in range(self.card_s):
                for a in range(self.card_a):
                    normalization = np.sum(self.P_estimation_0[h, s, a, :])
                    for ns in range(self.card_s):
                        self.P_estimation_0[h, s, a, ns] /= normalization
        self.P_estimation = self.P_estimation_0 
        self.uncertainty = np.zeros_like(self.tot)
        self.utility_array_estimation = []
        A_total_init = np.random.random((self.N, self.hor, self.card_s, self.L)) #initialize a random utiltiy estimator
        for i in range(self.N):
            self.utility_array_estimation.append(utilityFunc(self.L, self.card_s, A_total_init[i]))
        
        self.pi = np.ones((self.hor, self.card_s, self.card_a)) * (1./self.card_a) #initialize a uniform planner policy
        self.nu = agentPolicy(self.L, self.N, self.card_s, self.T, self.utility_array_estimation)
        self.nu_star = agentPolicySample(self.L, self.N, self.card_s, self.nu, num_sample, self.env)

        self.bid_init = np.zeros((self.N, self.L))
        for i in range(self.N):
            for j in range(self.L):
                self.bid_init[i][j] = 1 / self.N

    def _pre_train(self, M):
        for s in range(self.card_s):
            x_buffer_S = []
            u_buffer_S = []
            for h in range(self.hor):
                x_buffer_S_H = []
                u_buffer_S_H = []
                for t in range(M):
                    for j in range(self.L):
                        x = np.zeros((self.N, self.L))
                        u = np.zeros((self.N,))
                        for i in range(self.N):
                            x[i][j] = 1
                        u = self.env._utility_observation(h, s, x) 
                        x_buffer_S_H.append(x)
                        u_buffer_S_H.append(u)
                x_buffer_S.append(x_buffer_S_H)
                u_buffer_S.append(u_buffer_S_H)
            self.x_buffer.append(x_buffer_S)
            self.u_buffer.append(u_buffer_S)

        self._estimation_U()

    def _estimation_P(self):
        for h in range(self.hor):
            for s in range(self.card_s):
                for a in range(self.card_a):
                    for s_next in range(self.card_s):
                        self.P_estimation[h, s, a, s_next] = self.vis[h, s, a, s_next] / max(self.tot[h, s, a], 1) 
                    self.uncertainty[h, s, a] = self.hor * self.card_s * np.sqrt(np.log(2 * self.hor * self.card_s * self.card_a * self.card_s / self.delta) / max(self.tot[h, s, a], 1))
        
    def _estimation_U(self):
        self._Lambda = np.zeros((self.card_s, self.N, self.hor, self.L, self.L))
        self._R = np.zeros((self.card_s, self.N, self.hor, self.L))
        for s in range(self.card_s):
            for i in range(self.N):
                for h in range(self.hor):
                    for t in range(len(self.x_buffer[s][h])):
                        self._Lambda[s][i][h] += np.outer(self.x_buffer[s][h][t][i], self.x_buffer[s][h][t][i])
                        self._R[s][i][h] += self.x_buffer[s][h][t][i] * self.u_buffer[s][h][t][i]
                    self._A_total_bar[i][h][s] = np.matmul(np.linalg.inv(self._Lambda[s][i][h]), self._R[s][i][h].transpose())
                    self._A_total_opt[i][h][s] = np.random.multivariate_normal(self._A_total_bar[i][h][s].reshape((self.L,)), self.alpha * np.linalg.inv(self._Lambda[s][i][h]))

        for i in range(self.N):
            self.utility_array_estimation[i] = utilityFunc(self.L, self.card_s, self._A_total_opt[i])
        
    def _estimation_U_running(self):
        for s in range(self.card_s):
            for i in range(self.N):
                for h in range(self.hor):
                    self._Lambda[s][i][h] += np.outer(self.x_buffer[s][h][-1][i], self.x_buffer[s][h][-1][i])
                    self._R[s][i][h] += self.x_buffer[s][h][-1][i] * self.u_buffer[s][h][-1][i]
                    self._A_total_bar[i][h][s] = np.matmul(np.linalg.inv(self._Lambda[s][i][h]), self._R[s][i][h].transpose())
                    self._A_total_opt[i][h][s] = np.random.multivariate_normal(self._A_total_bar[i][h][s].reshape((self.L,)), self.alpha * np.linalg.inv(self._Lambda[s][i][h]))

        for i in range(self.N):
            self.utility_array_estimation[i] = utilityFunc(self.L, self.card_s, self._A_total_opt[i])
        
    def _P_loss(self):
        diff = self.env.p - self.P_estimation
        return np.linalg.norm(diff)

    def train_policy(self, train_num, alpha):
       
        # Initialization
        self._pre_train(10)
        Q = np.zeros([self.hor, self.card_s, self.card_a])
        nu_pi_perform, nu_random_perform, nu_piopt_perform, nuopt_pi_perform, nuopt_pioptnuopt_perform = [], [], [], [], []
        R_planner, R_random, R_total = [], [], []
        Regret_planner = 0
        Regret_total = 0
        Regret_random = 0
        
        for k in range(train_num):
            # Policy Improvement
            self.nu.update_utility_array(self.utility_array_estimation)
            for h in range(self.hor):
                for s in range(self.card_s):
                    tmp = 0.0  # normalization constant
                    for a in range(self.card_a):
                        self.pi[h, s, a] *= np.exp((alpha / np.sqrt(k+1)) * 0.05 * Q[h,s,a])
                        tmp += self.pi[h, s, a]
                    for a in range(self.card_a):
                        self.pi[h,s,a] /= tmp
 
            # Sample
            state = np.random.choice(range(self.card_s))
            utility_sample = np.zeros((self.N))
            for h in range(self.hor):
                action = np.random.choice(range(self.card_a), p=[self.pi[h,state,a] for a in range(self.card_a)])
                x, p = self.nu.take_action(h, state, self.bid_init)
                next_state, utility_obs, _ = self.env.step(h, state,action, x)
                self.vis[h, state, action, next_state] += 1
                self.tot[h, state, action] += 1
                self.x_buffer[state][h].append(x)
                self.u_buffer[state][h].append(utility_obs)
                state = next_state
                utility_sample += utility_obs
            
            # Policy Evaluation
            self._estimation_P()
            print('Epoch:', k, 'P Loss:', self._P_loss())
            self._estimation_U_running()

            Q = np.zeros([self.hor, self.card_s, self.card_a])
            V = np.zeros([self.hor, self.card_s])
            u = np.zeros([self.N])

            for h in reversed(range(self.hor)):
                for s in range(self.card_s):
                    x, p = self.nu.take_action(h, s, self.bid_init)
                    for i in range(self.N):
                        u[i] = self.utility_array_estimation[i].eval(h, s, x[i])
                    u_sum = np.sum(u)
                    for a in range(self.card_a):
                        PV = 0
                        if h < self.hor-1:
                            for s_ in range(self.card_s):
                                PV += self.P_estimation[h,s,a,s_] * V[h+1,s_]
                        Q[h,s,a] = np.clip(u_sum + self.kappa * self.uncertainty[h,s,a] + PV, 0, 10000)
                        V[h,s] += self.pi[h,s,a] * Q[h,s,a]

            # Evaluation
            if k % self.freq == 0:
                value_nu_pi = self.env._eval_new(self.pi, self.nu)
                pi_random = np.ones((self.hor, self.card_s, self.card_a)) * (1./self.card_a)
                value_nu_random = self.env._eval_new(pi_random, self.nu)
                pi_opt, value_nu_piopt = self.env._optimal_regulation_policy(self.nu)
                pi_opt_nuopt, value_nuopt_pioptnuopt = self.env._optimal_regulation_policy(self.nu_star)

                print('Social Welfare of Episode ', k, ', (nu, random):',value_nu_random)
                print('Social Welfare of Episode ', k, ', (nu, pi):',  value_nu_pi)
                print('Social Welfare of Episode ', k, ', (nu, pi*(nu)):',  value_nu_piopt)
                print('Social Welfare of Episode ', k, ', (nu*(nu), pi*(nu*(nu))):',  value_nuopt_pioptnuopt)

                Regret_planner += value_nu_piopt - value_nu_pi
                Regret_random += value_nuopt_pioptnuopt - value_nu_random
                Regret_total += value_nuopt_pioptnuopt - value_nu_pi

                nu_pi_perform.append(value_nu_pi)
                nu_random_perform.append(value_nu_random)
                nu_piopt_perform.append(value_nu_piopt)
                nuopt_pioptnuopt_perform.append(value_nuopt_pioptnuopt)
                R_planner.append(Regret_planner)
                R_random.append(Regret_random)
                R_total.append(Regret_total)

        return nu_pi_perform, nu_random_perform, nu_piopt_perform, nuopt_pioptnuopt_perform, R_planner, R_random, R_total

