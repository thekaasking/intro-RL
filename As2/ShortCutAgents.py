import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        if np.random.random() < 1 - self.epsilon + self.epsilon/self.n_actions:
            return np.argmax(self.Q[state,:])
        else:
            actions = [0,1,2,3]
            action = np.random.choice(actions, 1)
            maxaction = np.argmax(self.Q[state,:])
            while action==maxaction:
                action = np.random.choice(actions, 1)
            return action
        
    def update(self, state, action, reward, alpha, newstate):
        self.Q[state][action] = self.Q[state][action] + alpha * (reward+np.max(self.Q[newstate,:])-self.Q[state][action])


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = 1
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        action = np.argmax(self.Q[state,:])
        if np.random.random() > self.epsilon:
            return action
        else:
            a = np.random.choice(range(self.Q[state,:].size))
            while a == action:
                a = np.random.choice(range(self.Q[state,:].size))
            action = a
        return action
        
    def update(self, state, action, reward, alpha, state_1, action_1):
        """
        :param state_1: state+1
        :param action_1: action+1
        """
        self.Q[state][action] = self.Q[state][action] + alpha*(
        reward + (self.gamma*self.Q[state_1][action_1])-self.Q[state][action])


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass