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
#   Q(A) <- Q(A) + 1/N(A)[R - Q(A)] or
#   Q(A) <- Q(A) + a * [R - Q(A)]

# K-armed bandit class
# Creates a k-armed bandit instance. Takes in the number of arms, the selected epsilon value, and an optional alpha
# parameter for the constant step-size update approach
class Bandit:
    """
    K-armed bandit class. Inspired by the pseudocode and ideas
    from Sutton and Barto Chapter 3.
    """
    def __init__(self, k, epsilon, alpha=None):
        """
        Creates a k-armed bandit instance.

        :param k: The arms for the bandit
        :param epsilon: The epsilon value for e-greedy action selection
        :param alpha: The alpha value for step-size action value updates, is None by default.
        """
        self.k = k
        self.R = 0.0
        self.epsilon = epsilon
        self.alpha = alpha
        # Uncomment the line below for randomly initialized action table values
        # self.action_table = {i: (np.random.normal(loc=0.0, scale=1.0), 1.0) for i in range(k)}

        # Uncomment the lines below for equally initialized action table values (for random walks)
        self.qStar, self.sd = 0.0, 1.0
        self.action_table = {i: (self.qStar, self.sd) for i in range(k)}

        self.optimal = max(self.action_table, key=self.action_table.get)
        self.q = [0] * k
        self.n = [0] * k

    def generateReward(self, action) -> float:
        """
        Generates the reward for the selected bandit arm.

        :param action: The selected action by the bandit
        :return: The generated reward for the action
        """
        mean, sd = self.action_table[action]
        return np.random.normal(loc=mean, scale=sd)

    def updateValue(self, a, reward):
        """
        Updates the value of an arm given the current reward

        :param a: The arm to be updated
        :param reward: The most recent reward from pulling the arm
        :return:
        """
        self.n[a] += 1
        if self.alpha is None:
            self.q[a] = self.q[a] + (1. / float(self.n[a])) * (reward - self.q[a])
        else:
            self.q[a] = self.q[a] + self.alpha * (reward - self.q[a])

        # Uncomment the for loop to implement random walks
        for i in self.action_table:
            self.action_table[i] = (self.action_table[i][0] + np.random.normal(loc=0.0, scale=0.01),
                                    self.action_table[i][1])

        self.optimal = max(self.action_table, key=self.action_table.get)


def getInput():
    """
    Prompts the users for the required input for the simulation and returns
    it in a tuple.
    :return: The input from the user
    """
    print("How many arms will your bandits have?:")
    arms = int(input())
    print("What value(s) of epsilon do you want to use?:")
    epsilons = [float(x) for x in input().split(',')]
    print(epsilons)
    print("Do you want to use a constant step size parameter, alpha? [y/n]:")
    step = input()
    alpha = None
    if step == 'y':
        print("What value do you want for alpha?")
        alpha = float(input())
    print("How many bandits do you want per epsilon trial?:")
    bandits = int(input())
    print("How many iterations do you want per bandit?:")
    iterations = int(input())
    print("Running bandit simulation...")
    return arms, epsilons, alpha, bandits, iterations


def main():
    # Getting input from the user to create the simulation
    arms, epsilons, alpha, bandits, iterations = getInput()

    # Creating matrices for the rewards and optimal actions on each trial
    rewards = {0.0: np.zeros(shape=(bandits, iterations)),
               0.1: np.zeros(shape=(bandits, iterations)),
               0.01: np.zeros(shape=(bandits, iterations))}
    optimal_actions = {0.0: np.zeros(shape=(bandits, iterations)),
                       0.1: np.zeros(shape=(bandits, iterations)),
                       0.01: np.zeros(shape=(bandits, iterations))}
    figure, axes = plt.subplots(2, 1)

    """
    Main driver loop. Loops over epsilon values, bandits, and simulation
    iterations in that order. At each time step:
     * selects an action
     * generates a reward
     * updates the value function
    """
    for k in epsilons:
        for j in range(bandits):
            np.random.seed(j)
            b = Bandit(arms, k, alpha)
            for i in range(iterations):
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
        avg_rewards = rewards[k].mean(axis=0)
        avg_optimal = optimal_actions[k].mean(axis=0)
        axes[0].plot(range(iterations), avg_rewards, label="Epsilon = " + str(k))
        axes[1].plot(range(iterations), avg_optimal, label="Epsilon = " + str(k))

    # Creating plots for results
    axes[0].legend()
    axes[0].set_ylabel("Average Reward")
    axes[1].legend()
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("% Optimal Action")
    if alpha:
        figure.suptitle("Average Rewards and Optimal Action Selection in a non-stationary environment, " +
                        "using the constant step-size method, alpha = {}".format(alpha))
    else:
        figure.suptitle("Average Rewards and Optimal Action Selection in a non-stationary environment, " +
                        "using the sample average method".format(epsilons))
    plt.show()


if __name__ == '__main__':
    main()
