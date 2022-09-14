import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# A simple bandit algorithm:
# Initialize, for a=1 to k:
#   Q(a) <- 0
#   N(a) <- 0
# Loop forever:
#   A <- { argmax_a Q(a) with probability 1 - epsilon, a random action with probability epsilon
#   R <- bandit(A)
#   N(A) <- N(A) + 1
#   Q(A) <- Q(A) + 1/N(A)[R - Q(A)]


class Bandit:
    def __init__(self, k, epsilon):
        self.k = k
        self.R = 0.0
        self.epsilon = epsilon
        self.action_table = {i: (np.random.normal(loc=0.0, scale=1.0), 1.0) for i in range(k)}
        self.optimal = max(self.action_table, key=self.action_table.get)
        self.q = [0 for i in range(k)]
        self.n = [0 for i in range(k)]

    def generateReward(self, action) -> float:
        mean, sd = self.action_table[action]
        return np.random.normal(loc=mean, scale=sd)

    def updateValue(self, a, reward):
        self.n[a] += 1
        self.q[a] = self.q[a] + (1. / float(self.n[a])) * (reward - self.q[a])

def main():
    rewards = {0.0: np.zeros(shape=(2000, 1000)),
               0.1: np.zeros(shape=(2000, 1000)),
               0.01: np.zeros(shape=(2000, 1000))}
    optimal_actions = {0.0: np.zeros(shape=(2000, 1000)),
                       0.1: np.zeros(shape=(2000, 1000)),
                       0.01: np.zeros(shape=(2000, 1000))}
    s = 0
    for k in [0.0, 0.1, 0.01]:
        np.random.seed(s)
        for j in range(2000):
            b = Bandit(10, k)
            i = 0
            for i in range(1000):
                if np.random.rand() > b.epsilon:
                    A = np.argmax(b.q)
                else:
                    A = np.random.randint(0, b.k)
                if A == b.optimal:
                    optimal_actions[k][j][i] = 1
                R = b.generateReward(A)
                b.updateValue(A, R)
                b.R += R
                rewards[k][j][i] = R
        # avg_rewards = {0.0: rewards[0.0].mean(axis=0),
        #                0.1: rewards[0.1].mean(axis=0),
        #                0.01: rewards[0.01].mean(axis=0)}
        avg_optimal = {0.0: optimal_actions[0.0].mean(axis=0),
                       0.1: optimal_actions[0.1].mean(axis=0),
                       0.01: optimal_actions[0.01].mean(axis=0)}
        # plt.plot(range(1000), avg_rewards[k], label=str(k))
        plt.plot(range(1000), avg_optimal[k], label=str(k))
        s += 1
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
