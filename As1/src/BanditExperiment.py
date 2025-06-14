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
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth
 
def run_repetitions(n_actions, n_timesteps, epsilon):
    env = BanditEnvironment(n_actions=n_actions)
    i = 0
    y = np.zeros(n_timesteps)
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    for i in range(n_timesteps):
        a = pi.select_action(epsilon) # select action
        r = env.act(a) # sample reward
        pi.update(a,r) # update policy
        y[i] = r
        i += 1
    return y

def run_repetitions_oi(n_actions, n_timesteps, initial_value):
    """
    Adjusted 'run_repetitions' function for Optimistic Initialization
    Changes:
    -Removed epsilon parameter 
    -Added initial_value parameter
    """
    env = BanditEnvironment(n_actions=n_actions)
    i = 0
    y = np.zeros(n_timesteps)
    pi = OIPolicy(n_actions=n_actions, initial_value=initial_value) # Initialize policy
    for i in range(n_timesteps):
        a = pi.select_action() # select action
        r = env.act(a) # sample reward
        pi.update(a,r) # update policy
        y[i] = r
        i += 1
    return y

def run_repetitions_ucb(n_actions, n_timesteps, c):
    """
    Adjusted 'run_repetitions' function for UBC
    Changes:
    -Removed epsilon parameter 
    -Added c parameter
    """
    env = BanditEnvironment(n_actions=n_actions)
    i = 0
    y = np.zeros(n_timesteps)
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    for i in range(n_timesteps):
        a = pi.select_action(c=c, t=i) # select action
        r = env.act(a) # sample reward
        pi.update(a,r) # update policy
        y[i] = r
        i += 1
    return y


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    
    # Assignment 1: e-greedy ---------------------------------------------------
    LC_egreedy = LearningCurvePlot(title="Learning Curve epsilon-greedy policy")
    epsilon_values = [0.01, 0.05, 0.1, 0.25]
    eG_results_dict = dict()
    
    for epsilon in epsilon_values:
        eG_results = np.zeros((n_repetitions, n_timesteps))
        eG_results_dict[epsilon] = eG_results
        
        for i in range(n_repetitions):
           eG_results[i] = run_repetitions(n_actions, n_timesteps, epsilon)
           i+=1
        eG_avgs = np.sum(eG_results, axis = 0)/n_repetitions
        LC_egreedy.add_curve(smooth(eG_avgs,smoothing_window),label=f'epsilon={epsilon}, smoothed')

    LC_egreedy.save(name='learning_curve_egreedy.png')
    # --------------------------------------------------------------------------
    
    # Assignment 2: Optimistic init --------------------------------------------
    initial_values = [0.1, 0.5, 1.0, 2.0]
    OI_results_dict = dict()
    LC_OI = LearningCurvePlot(title="Learning Curve optimistic initialization policy")
    
    for initial_value in initial_values:
        OI_results = np.zeros((n_repetitions, n_timesteps))
        OI_results_dict[initial_value] = OI_results
        
        for i in range(n_repetitions):
            OI_results[i] = run_repetitions_oi(n_actions, n_timesteps, initial_value)
            i+=1
        OI_avgs = np.sum(OI_results, axis = 0)/n_repetitions
        LC_OI.add_curve(smooth(OI_avgs,smoothing_window),label=f"initial_value={initial_value}, smoothed")

    LC_OI.save(name='learning_curve_optimistic_init.png')
    # --------------------------------------------------------------------------  
    
    # Assignment 3: UCB
    LC_UCB = LearningCurvePlot(title="Learning Curve UCB policy")
    c_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    UCB_results_dict = dict()
    
    for c in c_values:
        UCB_results = np.zeros((n_repetitions, n_timesteps))
        UCB_results_dict[c]=UCB_results
        
        for i in range(n_repetitions):
            UCB_results[i] = run_repetitions_ucb(n_actions, n_timesteps, c)
            i+=1
        UCB_avgs = np.sum(UCB_results, axis = 0)/n_repetitions
        LC_UCB.add_curve(smooth(UCB_avgs,smoothing_window),label=f'c={c}, smoothed')

    LC_UCB.save(name='learning_curve_UCB.png')
    # --------------------------------------------------------------------------

    # Assignment 4: Comparison
    # A
    CPlot = ComparisonPlot(title='Approach comparison egreedy, OI, UCB policies')
    
    # sum of [sum of [rewards] for all timesteps] for all repetitions
    # calculate the mean reward for the three approaches and their different values
    mean_reward_eG = np.zeros(len(epsilon_values))
    mean_reward_OI = np.zeros(len(initial_values))
    mean_reward_UCB = np.zeros(len(c_values))
    
    i = 0
    for epsilon, rewards in eG_results_dict.items():
        mean_reward_eG[i] = np.sum(rewards)/(n_repetitions*n_timesteps)
        i += 1
    
    i = 0
    for initial_value, rewards in OI_results_dict.items():    
        mean_reward_OI[i] = np.sum(rewards)/(n_repetitions*n_timesteps)
        i += 1

    i = 0
    for c, rewards in UCB_results_dict.items():
        mean_reward_UCB[i] = np.sum(rewards)/(n_repetitions*n_timesteps)
        i += 1
    
    # Add the three curves
    CPlot.add_curve(epsilon_values,mean_reward_eG,label="e-greedy")
    CPlot.add_curve(initial_values,mean_reward_OI,label="Optimistic Initialization")
    CPlot.add_curve(c_values,mean_reward_UCB,label="UCB")
    
    CPlot.save(name="approach_comparison.png")
    
    # B
    # Best values, derived from the generated plot above
    epsilon_optimal = 0.05
    initial_value_optimal = 0.5
    c_optimal = 0.25
    
    LC_comparison = LearningCurvePlot(title="Optimal values reward comparison")
    
    # e-greedy curve
    eG_results = np.zeros((n_repetitions, n_timesteps))
    
    for i in range(n_repetitions):
        eG_results[i] = run_repetitions(n_actions, n_timesteps, epsilon_optimal)
        i+=1
    eG_avgs = np.sum(eG_results, axis = 0)/n_repetitions
    LC_comparison.add_curve(smooth(eG_avgs,smoothing_window),label=f'e-greedy, epsilon={epsilon_optimal}, smoothed')
    
    # IO curve
    OI_results = np.zeros((n_repetitions, n_timesteps))        
    for i in range(n_repetitions):
        OI_results[i] = run_repetitions_oi(n_actions, n_timesteps, initial_value_optimal)
        i+=1
    OI_avgs = np.sum(OI_results, axis = 0)/n_repetitions
    LC_comparison.add_curve(smooth(OI_avgs,smoothing_window),label=f"IO, initial_value={initial_value_optimal}, smoothed")
    
    # UCB curve
    UCB_results = np.zeros((n_repetitions, n_timesteps))
    for i in range(n_repetitions):
        UCB_results[i] = run_repetitions_ucb(n_actions, n_timesteps, c_optimal)
        i+=1
    UCB_avgs = np.sum(UCB_results, axis = 0)/n_repetitions
    LC_comparison.add_curve(smooth(UCB_avgs,smoothing_window),label=f'UCB, c={c_optimal}, smoothed')

    LC_comparison.save(name='learning_curves_optimal_comparison.png')
    # --------------------------------------------------------------------------
    
    

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)
