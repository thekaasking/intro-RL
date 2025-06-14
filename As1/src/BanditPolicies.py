#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment

class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)
        
    def select_action(self, epsilon):
        action = np.argmax(self.Q)
        if np.random.random() > epsilon:
            return action
        else:
            a = np.random.choice(range(self.Q.size))
            while a == action:
                a = np.random.choice(range(self.Q.size))
            action = a
        return action
        
    def update(self,a,r):
        self.n[a]+=1
        self.Q[a] = self.Q[a]+((r-self.Q[a])/self.n[a])

class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0):
        self.n_actions = n_actions
        self.Q = np.full(n_actions, initial_value)
        
    def select_action(self):
        action = np.argmax(self.Q)
        # kansen gaan zetten op de acties
        # kans van 1 op de beste actie zetten, de rest 0
        action = np.argmax(self.Q)
        return action
        
    def update(self,a,r):
        alpha = 0.1 # possibly move this elsewhere?
        self.Q[a] = self.Q[a]+(alpha*(r-self.Q[a]))

class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)
    
    def select_action(self, c, t):
        if 0 in self.n:
            return np.where(self.n == 0)[0][0]
        action = np.argmax(self.Q+c*np.sqrt(np.log(t)/self.n))
        return action
        
    def update(self,a,r):
        self.n[a]+=1
        self.Q[a] = self.Q[a]+((r-self.Q[a])/self.n[a])
    
def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
    
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(epsilon=0.5) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    
    pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    a = pi.select_action() # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a,r))
    
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(c=1.0,t=1) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test UCB policy with action {}, received reward {}".format(a,r))
    
if __name__ == '__main__':
    test()
