import numpy as np


class utilityFunc:
    def __init__(self, L, C_card, A):
        self.L = L
        self.C_card = C_card
        self.A = A  #shape of A: (H, C_card, L)

    def eval(self, h, c, x):   #shape of x: (L,)
        return np.inner(self.A[h][c], x)

    def observation(self, h, c, x):
        return np.inner(self.A[h][c], x) + np.random.normal(0, 0.001)


class utilityFuncFixC(utilityFunc):
    def __init__(self, L, C_card, A, c):
        utilityFunc.__init__(self, L, C_card, A)
        self.context = c

    def eval_c(self, x):
        return self.eval(self.context, x)


def PRD_find_CE(utility_array, N, L, h, c, bid_init, T):
    assert N==L, "N does not equal to L!"

    bid = np.zeros((N, L)) #shape of bid: (N, L)
    for i in range(N):
            for j in range(L):
                bid[i][j] = 1 / N
    p = np.zeros(L)
    u = np.zeros(N)
    x = np.zeros_like(bid)

    for t in range(T):
        for i in range(N):
            for j in range(L):
                if bid[i][j] == 0:
                    x[i][j] = 0
                else:
                    x[i][j] = bid[i][j]/np.sum(bid, axis=0)[j]
            u[i] = utility_array[i].eval(h, c, x[i])
        p = np.sum(bid, axis=0)

        for i in range(N):
            for j in range(L):
                bid[i][j] = utility_array[i].A[h][c][j] * x[i][j] * p[i] / u[i]

    return x, p


class agentPolicy:
    def __init__(self, L, N, C_card, T, utility_array):
        self.L = L
        self.N = N
        self.C_card = C_card
        self.T = T
        self.utility_array = utility_array

    def update_utility_array(self, utility_array_new):
        self.utility_array = utility_array_new

    def take_action(self, h, c, bid_init):
        return PRD_find_CE(self.utility_array, self.N, self.L, h, c, bid_init, self.T)


class agentPolicySample:
    def __init__(self, L, N, C_card, nu, num_sample, env):
        self.L = L
        self.N = N
        self.C_card = C_card
        self.nu = nu
        self.bid_init = []
        self.num_sample = num_sample
        self.env = env
        assert self.L == 3 and self.N == 3

    def _CE_loss(self, h, c, x, x_prime, p):
        loss = 0
        for i in range(self.N):
            loss += self.env.utility_array_true[i].eval(h, c, x_prime[i]) \
                - self.env.utility_array_true[i].eval(h, c, x[i])
        return loss

    def take_action(self, h, c, bid_init):
        x_nu, p_nu = self.nu.take_action(h, c, [])
        loss_array = np.zeros((self.num_sample))
        x_array = np.zeros((self.num_sample, self.N, self.L))

        for t in range(self.num_sample):
            budget_flag = 1
            sample = np.random.rand(4)
            x = np.zeros_like(x_nu)
            x[0,1] = sample[0]
            x[0,2] = sample[1]
            x[1,1] = sample[2]
            x[1,2] = sample[3]
            x[0,0] = (x_nu[0,0]*p_nu[0]+x_nu[0,1]*p_nu[1]+x_nu[0,2]*p_nu[2]-x[0,1]*p_nu[1]-x[0,2]*p_nu[2])/p_nu[0]
            x[1,0] = (x_nu[1,0]*p_nu[0]+x_nu[1,1]*p_nu[1]+x_nu[1,2]*p_nu[2]-x[1,1]*p_nu[1]-x[1,2]*p_nu[2])/p_nu[0]
            for _ in range(3):
                x[2,_] = 1-x[0,_]-x[1,_]
            x_array[t] = x

            for i in range(self.N):
                if np.inner(x_array[t, i], p_nu) <= np.inner(x_nu[i], p_nu) + 0.0001 and (x_array[t, i]>=0).all():
                    budget_flag = 1
                else:
                    budget_flag = 0
                    break
            if budget_flag == 0:
                loss_array[t] = -1
                continue
            loss_array[t] = self._CE_loss(h, c, x_nu, x_array[t], p_nu)

        index = np.argmax(loss_array)
        if loss_array[index] <= 0:
            return x_nu, p_nu
        else:
            return x_array[index], p_nu
            