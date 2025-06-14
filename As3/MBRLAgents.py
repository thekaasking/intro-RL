#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.n = np.zeros((n_states, n_actions, n_states))
        self.r_sum = np.zeros((n_states, n_actions, n_states))
        self.reward = np.zeros((n_states, n_actions, n_states))
        self.p = np.zeros((n_states, n_actions, n_states))
        
    def select_action(self, s, epsilon):
        if np.random.random() < 1 - epsilon + epsilon/self.n_actions:
            return np.argmax(self.Q_sa[s,:])
        else:
            actions = [0,1,2,3]
            action = np.random.choice(actions, 1)
            maxaction = np.argmax(self.Q_sa[s,:])
            while action==maxaction:
                action = np.random.choice(actions, 1)
        return int (action)
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        if not done:
            self.n[s][a][s_next] += 1
            self.r_sum[s][a][s_next] += r
            self.p[s][a][s_next] = self.n[s][a][s_next]/np.sum(self.n[s, a, :])
            self.reward[s][a][s_next] = self.r_sum[s][a][s_next]/self.n[s][a][s_next]
            self.Q_sa[s][a] += self.learning_rate*(r+self.gamma*np.max(self.Q_sa[s_next, :])-self.Q_sa[s][a])

            for i in range(n_planning_updates):
                if not done:
                    state = np.random.choice(self.n_states)

                    while np.sum(self.n[state,:,:])==0:
                        state = np.random.choice(self.n_states)
                    action = np.random.choice(self.n_actions)

                    while np.sum(self.n[state, action, :])==0:
                        action = np.random.choice(self.n_actions)
                    state_next = np.argmax(self.p[state, action, :])

                    reward = self.reward[state][action][state_next]
                    self.Q_sa[state][action] += self.learning_rate*(reward+self.gamma*np.max(self.Q_sa[state_next, :])-self.Q_sa[state][action])


class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        self.predecessors = dict()

        self.Q_sa = np.zeros((n_states,n_actions))
        self.n = np.zeros((n_states, n_actions, n_states))
        self.r_sum = np.zeros((n_states, n_actions, n_states))
        self.reward = np.zeros((n_states, n_actions, n_states))
        self.p = np.zeros((n_states, n_actions, n_states))

    def select_action(self, s, epsilon):
        if np.random.random() < 1 - epsilon + epsilon/self.n_actions:
            return np.argmax(self.Q_sa[s,:])
        else:
            actions = [0,1,2,3]
            action = np.random.choice(actions, 1)
            maxaction = np.argmax(self.Q_sa[s,:])
            while action==maxaction:
                action = np.random.choice(actions, 1)
        return int (action)
        
    
    def update(self,s,a,r,done,s_next,n_planning_updates):
        if not done:
            # Update model
            self.n[s][a][s_next] += 1 # Update transition counts
            self.r_sum[s][a][s_next] += r # update reward sums 
            self.p[s][a][s_next] = self.n[s][a][s_next]/np.sum(self.n[s, a, :]) # Estimate transition function
            self.reward[s][a][s_next] = self.r_sum[s][a][s_next]/self.n[s][a][s_next] # Estimate reward function
            
            if s_next not in self.predecessors:
                self.predecessors[s_next] = [(s, a)]
            else:
                for i, (prev_s, prev_a) in enumerate(self.predecessors[s_next]):
                    if prev_s == s and prev_a == a:
                        self.predecessors[s_next][i] = (s, a)
                        break
                else:
                    self.predecessors[s_next].append((s, a))

            # Compute priority
            priority = abs(self.reward[s][a][s_next] +self.gamma*np.max(self.Q_sa[s_next, :])-self.Q_sa[s][a])
            
            if priority > self.priority_cutoff: # Compare priority
                self.queue.put((-priority,(s,a))) # State-action needs update
            
            for i in range(n_planning_updates):
                if not done:

                    # skip forloop iteration if the queue is empty
                    if not self.queue.empty():
                        top_p = self.queue.get() # get the top (priority(s,a)) for the queue
                        state, action = top_p[1]

                        # Simulate model
                        state_next = np.argmax(self.p[state, action, :])
                        reward = self.reward[state][action][state_next]

                        # Update Q table
                        self.Q_sa[state][action] += self.learning_rate*(reward+self.gamma*np.max(self.Q_sa[state_next, :])-self.Q_sa[state][action])                     
                        
                        # loop for all state, action predicted lead to _state
                        if state not in self.predecessors.keys():
                            continue

                        for s_new, a_new in self.predecessors[state]:
                            if self.n[s_new][a_new][state] > 0:
                                # calc reward
                                reward = self.reward[s_new][a_new][state]
                                
                                priority = abs(reward +self.gamma*np.max(self.Q_sa[state, :])-self.Q_sa[s_new][a_new])
                                if priority > self.priority_cutoff:
                                    self.queue.put((-priority,(s_new,a_new)))  

            
def test():

    n_timesteps = 1000
    gamma = 0.99

    # Algorithm parameters
    policy = 'ps' # 'ps' 
    epsilon = 0.05
    learning_rate = 0.5
    n_planning_updates = 5

    # Plotting parameters
    plot = False
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
                        


if __name__ == '__main__':
    test()
