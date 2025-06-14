"""
Introduction to Reinforcement Learning Assignment 2
"Model-free Reinforcement Learning", LIACS, 2023.

Written by: Liva van der Velden & Razo van Berkel
""" 

# Libraries imports
import numpy as np
import scipy as sp
import matplotlib as pl # perhaps unneeded if weu use Helper
from Helper import LearningCurvePlot, ComparisonPlot, smooth, NewPlot
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import Environment, ShortcutEnvironment, WindyShortcutEnvironment


def run_repetitions(n_episodes, alpha, epsilon):
    """
    Function that repeatedly tests the agent QLearningAgent()
    Creates fresh agent and environment for each repetition.
    
    Makes a plot of the action with the maximum value for each state
    :param epsilon: epsilon value, fixed at 0.1 in main.
    """ 
    env = ShortcutEnvironment()
    y = np.zeros(env.state_size())
    rewards = np.zeros(n_episodes)
    pi = QLearningAgent(n_actions=env.action_size(), epsilon = epsilon, n_states = env.state_size())
    for i in range(n_episodes):
        env.reset()
        while not env.done():
            state = env.state()
            a = pi.select_action(state) # select action
            r = env.step(a) # sample reward
            y[state] = np.argmax(pi.Q[state,:])
            pi.update(state, a, r, alpha, env.state()) # update policy
            rewards[i] += r
    return y, rewards


def run_repetitions_SARSA(n_timesteps, alpha, epsilon):
    """
    Function that repeatedly tests the agent SARSAAgent()
    Creates fresh agent and environment for each repetition.
    
    :param n_episodes: amount of episoned 
    :param alpha: alpha value, fixed at 0.1 in main.
    :param epsilon: epsilon value, fixed at 0.1 in main.
    :return y: return array filled with reward values for 
    """ 
    env = ShortcutEnvironment()
    y = np.zeros(env.state_size())
    rewards = np.zeros(n_timesteps)
    pi = SARSAAgent(n_actions=env.action_size(), epsilon = epsilon, n_states = env.state_size())
    for i in range(n_timesteps):
        env.reset()
        while not env.done():
            state = env.state()
            a = pi.select_action(state) # select action
            r = env.step(a) # sample reward
            action_1 = pi.select_action(state)
            
            y[state] = np.argmax(pi.Q[state,:])
            pi.update(state, a, r, alpha, env.state(), action_1) # update policy
            rewards[i] += r
    return y, rewards


def experiment(n_timesteps, n_repetitions, smoothing_window, epsilon, alpha_values):
    """
    Function that runs the experiments with the different agent. 
    Tests for multiple alpha values.
    
    :param n_timesteps: amount of episoned 
    :param n_repetitions: amount of repetitions 
    :param alpha_values: alpha values, list.
    :param epsilon: epsilon value, fixed at 0.1 in main.
    :param smoothing_window: smoothing value for the plots
    """
    #To Do: Write all your experiment code here
    
    # Assignment 1: Q-learning ---------------------------------------------------
    LC_Qlearning = LearningCurvePlot(title = "Learning curve for Q-learning")
    size = 12*12
    for alpha in alpha_values:
        eG_results = np.zeros((n_repetitions, size))
        results = np.zeros((n_repetitions, n_timesteps))
        for i in range(n_repetitions):
            eG_results[i], results[i]  = run_repetitions(n_timesteps, alpha, epsilon)
        avgs = np.sum(results, axis = 0)/n_repetitions
        LC_Qlearning.add_curve(smooth(avgs,smoothing_window),label=f'alpha={alpha}, not smoothed')
    LC_Qlearning.save(name='learning_curve_Q_learning.png')

    # Assignment 2: SARSA ---------------------------------------------------
    LC_SARSA = LearningCurvePlot(title = "Learning curve for SARSA")
    size = 12*12
    for alpha in alpha_values:
        eG_results = np.zeros((n_repetitions, size))
        results = np.zeros((n_repetitions, n_timesteps))
        for i in range(n_repetitions):
            eG_results[i], results[i]  = run_repetitions_SARSA(n_timesteps, alpha, epsilon)
        avgs = np.sum(results, axis = 0)/n_repetitions
        LC_SARSA.add_curve(smooth(avgs,smoothing_window),label=f'alpha={alpha}, not smoothed')
    LC_SARSA.save(name='learning_curve_SARSA.png')


def generate_greedy_plot(arr_y):
    """
    Function that generates the plot and path of the agent from start
    to finish. Start pos is not indicated in plot, but printed in print_greedy_plot.
    
    :param arr_y: array of size env.r*env.c, generated from a run_repetitions
    """
    env = ShortcutEnvironment()
    env.render()
    action_map = {0: '^', 1: 'v', 2: '<', 3:'>'}
    col_counter = 0
    
    plot = [] # list to store all rows in the plot
    row = []  # temporary list to store all actions of a row

    for action in arr_y:
        #print(type(action))
        row.append(action_map[action])
        col_counter+=1
        if (col_counter % 12)==0:
            plot.append(row)
            row = []
            #col_counter = 0
    
    # Print the plot
    print_greedy_plot(plot, env)


def print_greedy_plot(plot, env):
    """
    Prints the greedy path plot.

    :param plot: 2d list containing the rows and 
    :param env: environment object
    """
    # Start symbol
    print(f"Startpos x: {env.x}")
    print(f"Startpos y: {env.y}")    
    # print the figure
    # note: startpos is not marked. have to mark in paint afterwads
    print("+=========================+")
    for row in plot:
        print("| ", end="")
        for move in row:
            print(move, end=" ")
        print("|")
    print("+=========================+")


def windy_experiment(n_episodes=10000, alpha=0.1, espilon=0.1):
    """
    Function that runs an experiment in the windy environment.
    Code modified from run_repetitions and greedy_plot.
    
    :param n_episodes: amount of episodes (fixed 10000)
    :param alpha: alpha value (fixed 0.1)
    :param epsilon: epsilon value (fixed 0.1)
    """
    # initiate the environment and run the repetitions
    env = WindyShortcutEnvironment()
    env.render()
    y = np.zeros(env.state_size())
    rewards = np.zeros(n_episodes)
    pi = QLearningAgent(n_actions=env.action_size(), epsilon = epsilon, n_states = env.state_size())
    for i in range(n_episodes):
        env.reset()
        while not env.done():
            state = env.state()
            a = pi.select_action(state) # select action
            r = env.step(a) # sample reward
            y[state] = np.argmax(pi.Q[state,:])
            pi.update(state, a, r, alpha, env.state()) # update policy
            rewards[i] += r
    
    # Generate and print the greedy path plot
    action_map = {0: '^', 1: 'v', 2: '<', 3:'>'}
    col_counter = 0
    plot = [] # list to store all rows in the plot
    row = []  # temporary list to store all actions of a row

    for action in y:
        #print(type(action))
        row.append(action_map[action])
        col_counter+=1
        if (col_counter % 12)==0:
            plot.append(row)
            row = []
    print_greedy_plot(plot, env)


if __name__ == '__main__':  
    # experiment settings
    smoothing_window = 31
    n_episodes = 10000
    n_rep = 100
    epsilon = 0.1
    alpha = 0.1
    alpha_values = [0.01, 0.1, 0.5, 0.9]

    np.set_printoptions(threshold=np.inf)
    windy_experiment(n_episodes, alpha, epsilon)
    
    array_y, rewards = run_repetitions(n_episodes, alpha, epsilon)
    print("** Q-learning **")
    generate_greedy_plot(array_y)

    #experiment(n_episodes, n_rep, smoothing_window, epsilon, alpha_values)
    

    #array_y, rewards = run_repetitions_SARSA(n_episodes, alpha, epsilon)  
    #print("\n** SARSA **")
    #generate_greedy_plot(array_y)