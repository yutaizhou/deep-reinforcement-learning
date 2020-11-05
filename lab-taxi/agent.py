import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.1
        self.eps = 1
        self.eps_decay = 0.9998

    def _update_eps(self, current_episode, total_episode): 
        # eps = 1 / current_episode
        # self.eps = eps if eps >= 0.1 else 0.1
        # if self.eps > 0.1:
        self.eps *= self.eps_decay 
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        greedy_action = np.argmax(self.Q[state])
        random_action = np.random.choice(self.nA)
        return greedy_action if np.random.uniform() > self.eps else random_action

    def step(self, state, action, reward, next_state, done, i_episode, num_episodes):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        target = reward + np.max(self.Q[next_state])
        current = self.Q[state][action]
        self.Q[state][action] += self.alpha * (target - current)

        self._update_eps(i_episode, num_episodes)
        
