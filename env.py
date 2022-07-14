import numpy as np
from ce import utilityFunc


class env:
    def __init__(self, n, card_s, card_a, N, L, hor, init_U_P = False):
        self.n = n
        self.card_s = card_s
        self.card_a = card_a
        self.N = N
        self.L = L
        self.hor = hor
        self.init_s = 0

        # TRANSITION
        if init_U_P:
            self.p = np.random.random((self.hor, self.card_s, self.card_a, self.card_s)) # shape: (H, S, A, S)
            for h in range(self.hor):
                for s in range(self.card_s):
                    for a in range(self.card_a):
                        self.p[h, s, a] = np.exp(self.p[h, s, a])
                        self.p[h, s, a] = np.exp(self.p[h, s, a])
                        self.p[h, s, a] = np.exp(self.p[h, s, a])
                        normalization = np.sum(self.p[h, s, a, :])
                        for ns in range(self.card_s):
                            self.p[h, s, a, ns] /= normalization
            np.save(file="transition_matirx.npy", arr=self.p)
        else:
            self.p = np.load(file="transition_matrix.npy")
        
        # UTILITY
        if init_U_P:
            self.A_total_true = np.random.random((self.N, self.hor, self.card_s, self.L)) # shape: (N, H, S, L)
            for i in range(self.N):
                for h in range(self.hor):
                    index = np.random.choice(range(self.card_s), 4, replace=False)
                    self.A_total_true[i, h, index[0]] = self.A_total_true[i, h, index[0]] * 0.1
                    self.A_total_true[i, h, index[1]] = self.A_total_true[i, h, index[1]] * 0.1
                    self.A_total_true[i, h, index[2]] = self.A_total_true[i, h, index[2]] * 10
                    self.A_total_true[i, h, index[3]] = self.A_total_true[i, h, index[3]] * 10
            np.save(file="utility_matrix.npy", arr=self.A_total_true)
        else:
            self.A_total_true = np.load(file="utility_matrix.npy")
        self.utility_array_true = []
        for i in range(self.N):
            self.utility_array_true.append(utilityFunc(self.L, self.card_s, self.A_total_true[i]))

    def _utility_observation_single_agent(self, i, h, s, x):
        return self.utility_array_true[i].observation(h, s, x)

    def _utility_observation(self, h, s, x):
        u = np.zeros((self.N,))
        for i in range(self.N):
            u[i] = self._utility_observation_single_agent(i, h, s, x[i])
        return u #shape: (N, )

    def _utility_mean_single_agent(self, i, h, s, x):
        return self.utility_array_true[i].eval(h, s, x)

    def _utility_mean(self, h, s, x):
        u = np.zeros((self.N,))
        for i in range(self.N):
            u[i] = self._utility_mean_single_agent(i, h, s, x[i])
        return u #shape: (N, )

    def step(self, h, s, a, x):
        s_next = np.random.choice(np.arange(self.card_s), p=self.p[h, s, a].reshape(-1))
        u = self._utility_observation(h, s, x)
        return s_next, u, None

    def _eval(self, pi, nu):
        bid_init = np.zeros((self.N, self.L))
        for i in range(self.N):
            for j in range(self.L):
                bid_init[i][j] = 1 / self.N

        prob_s = np.zeros((self.card_s))
        prob_s[self.init_s] = 1
        utility = np.zeros((self.N))
        for h in range(self.hor):
            prob_s_next = np.zeros([self.card_s])
            for s in range(self.card_s):
                for a in range(self.card_a):
                    for s_next in range(self.card_s):
                        prob_s_next[s_next] += prob_s[s] * pi[h, s, a] * self.p[h, s, a, s_next]
                x, p = nu.take_action(h, s, bid_init) 
                utility += prob_s[s] * self._utility_mean(h, s, x)
            prob_s = prob_s_next
        
        social_welfare = np.sum(utility)
        return utility, social_welfare

    def _eval_new(self, pi, nu):
        PV = np.zeros([self.hor, self.N, self.card_s, self.card_a])
        PV_sum = np.zeros([self.hor, self.card_s, self.card_a])
        V = np.zeros([self.hor, self.N, self.card_s])
        bid_init = np.zeros((self.N, self.L))
        for i in range(self.N):
            for j in range(self.L):
                bid_init[i][j] = 1 / self.N

        for h in reversed(range(self.hor)):
            for s in range(self.card_s):
                x, p = nu.take_action(h, s, bid_init)
                for i in range(self.N):
                    for a in range(self.card_a):
                        if h < self.hor-1:
                            for s_ in range(self.card_s):
                                PV[h, i, s, a] += self.p[h, s, a, s_] * V[h+1, i, s_]
                for a in range(self.card_a):
                    PV_sum[h, s, a] = np.sum(PV[h, :, s, a])
                for i in range(self.N):
                    for a in range(self.card_a):
                        V[h, i, s] += PV[h, i, s, a] * pi[h][s][a]
                    V[h, i, s] += self._utility_mean_single_agent(i, h, s, x[i])

        return np.sum(V[0, :, self.init_s])

    def _optimal_regulation_policy(self, nu):
        PV = np.zeros([self.hor, self.N, self.card_s, self.card_a])
        PV_sum = np.zeros([self.hor, self.card_s, self.card_a])
        V = np.zeros([self.hor, self.N, self.card_s])
        pi = np.zeros((self.hor, self.card_s, self.card_a))
        bid_init = np.zeros((self.N, self.L))
        for i in range(self.N):
            for j in range(self.L):
                bid_init[i][j] = 1 / self.N

        for h in reversed(range(self.hor)):
            for s in range(self.card_s):
                x, p = nu.take_action(h, s, bid_init)
                for i in range(self.N):
                    for a in range(self.card_a):
                        if h < self.hor-1:
                            for s_ in range(self.card_s):
                                PV[h, i, s, a] += self.p[h, s, a, s_] * V[h+1, i, s_]
                for a in range(self.card_a):
                    PV_sum[h, s, a] = np.sum(PV[h, :, s, a])
                pi[h][s][np.argmax(PV_sum[h, s])] = 1
                for i in range(self.N):
                    V[h, i, s] += self._utility_mean_single_agent(i, h, s, x[i]) + PV[h, i, s, np.argmax(pi[h][s])]
        
        return pi, np.sum(V[0, :, self.init_s])
