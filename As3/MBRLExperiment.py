#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent,PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, learning_rate, gamma,
                    epsilon, n_planning_updates):
    # Write all your experiment code here
    # Look closely at the code in test() of MBRLAgents.py for an example of the execution loop
    # Log the obtained rewards during a single training run of n_timesteps, and repeat this proces n_repetitions times
    # Average the learning curves over repetitions, and then additionally smooth the curve
    # Be sure to turn environment rendering off! It heavily slows down your runtime
    
    #learning_curve = np.random.rand(n_timesteps) # TO DO: replace this with a true experiment!
    plot = False
    plot_optimal_policy = True
    step_pause = 0.0001

    results = np.zeros((n_repetitions, n_timesteps))

    for k in range(n_repetitions):
        # Initialize environment and policy
        env = WindyGridworld()
        if policy == 'Dyna':
            pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
        elif policy == 'Prioritized Sweeping':    
            pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
        else:
            raise KeyError('Policy {} not implemented'.format(policy))
        
        # Prepare for running
        s = env.reset()  
        continuous_mode = True
        #result = np.zeros(n_timesteps)
        
        for t in range(n_timesteps):            
            # Select action, transition, update policy
            a = pi.select_action(s,epsilon)
            s_next,r,done = env.step(a)
            #if done:
            #    print("We zijn er")
            pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
            results[k][t] += r
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
        #results[k] = result
        #print(results)
    learning_curve = np.sum(results, axis = 0)/n_repetitions   
    # Apply additional smoothing
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve


def experiment():

    n_timesteps = 10000
    n_repetitions = 10
    smoothing_window = 500
    gamma = 0.99

    for policy in ['Dyna','Prioritized Sweeping']:
    
        ##### Assignment a: effect of epsilon ######
        learning_rate = 0.5
        n_planning_updates = 5
        epsilons = [0.01,0.05,0.1,0.25]
        Plot = LearningCurvePlot(title = '{}: effect of $\epsilon$-greedy'.format(policy))
        
        for epsilon in epsilons:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='$\epsilon$ = {}'.format(epsilon))
        Plot.save('{}_egreedy.png'.format(policy))
        ##### Assignment b: effect of n_planning_updates ######
        epsilon=0.05
        n_planning_updatess = [1,5,15]
        learning_rate = 0.5
        Plot = LearningCurvePlot(title = '{}: effect of number of planning updates per iteration'.format(policy))

        for n_planning_updates in n_planning_updatess:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='Number of planning updates = {}'.format(n_planning_updates))
        Plot.save('{}_n_planning_updates.png'.format(policy))  
        
        ##### Assignment c: effect of learning_rate ######
        epsilon=0.05
        n_planning_updates = 5
        learning_rates = [0.1,0.5,1.0]
        Plot = LearningCurvePlot(title = '{}: effect of learning rate'.format(policy))
    
        for learning_rate in learning_rates:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='Learning rate = {}'.format(learning_rate))
        Plot.save('{}_learning_rate.png'.format(policy)) 
    
if __name__ == '__main__':
    #learning_curve = run_repetitions('dyna', 10, 10000, 101, 0.5, 0.99, 0.05, 5)
    #Plot = LearningCurvePlot(title = 'testrun')
    #Plot.add_curve(learning_curve,label='testrun')
    #Plot.save(name = 'testrun.png')
    experiment()