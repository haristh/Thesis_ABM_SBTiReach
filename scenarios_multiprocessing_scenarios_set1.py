#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:38:53 2023

@author: charistheodorou

One of the three python files that focus on the experimentation of the thesis. 
This one presents the first set of scenarios i.e. the scenarios influencingdirectly the
pressures by multiplying their values by factors of 5 and 10. The model runs from 2015 
until 2025 (108 steps) with 2233 companies (total) 50 times and averages and stds 
are calculated.  Multiprocessing was used to reduce the time taken.

Running the data collection, saving figures and saving any data is commented out to avoid overwritin



"""

import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SBTiModel_main import SBTiModel 
import os
from multiprocessing import Pool

import time
#%%
start_time = time.time()


num_companies = 20
number_of_steps = 108

#network
alpha = 0.15
beta = 0.15
gamma = 0.15
delta = 0.05
rewiring_frequency = 13   # steps between rewiring function

sigma = 0.025
seed = None


#awareness
aware_update_step = 3               # aware due to interaction with other companies
companies_turn_aware_per_round = 1  # how often companies turn aware due to SBTi campaigns
steps_per_round = 7                 # agents randomly turn aware every these many steps, affected by SBTi


#Committing
meetings_per_year = 10    # average of existing data- used to check if a company will commit
pres_mot_eval = "product" # serial, sum, product

#CDPcampaign_pressure_factors= [1.0,1.56,1.33]
manufacturing_coefficient = 0.232
non_manufacturing_coefficient = 0.358
shareholder_pressure_coefficient = 0.253
manager_pressure_coefficient = 0.254
employee_pressure_coefficient = 0.238
market_pressure_coefficient = 0.210
#start_campaign = 52 
pressure_threshold = 9.02
motivation_threshold = 7.73


#setting a target
work_rate = 0.47
internal_target_range=[0.0, 0.5]
max_comm_duration = 24


#levers
leadership_lever = 1.0
risk_awareness_lever = 1.0
reputation_lever = 1.0         
manager_pressure_lever = 1.0     
shareholder_pressure_lever = 1.0
employee_pressure_lever= 1.0
market_pressure_lever = 1.0


#%%



scenarios = {
    'ShareholderPressure':  [(5,1,1,1), (10,1,1,1)],
    'ManagerPressure':      [(1,5,1,1), (1,10,1,1)],
    'EmployeePressure':     [(1,1,5,1), (1,1,10,1)],
    'MarketPressure':       [(1,1,1,5), (1,1,1,10)],
    'AllPressure':          [(5,5,5,5), (10,10,10,10)],
    'NoIncrease':           [(1,1,1,1)]
}

#%%
all_model_results = {}



def run_simulation(params):
    scenario, params_set, run_number = params
    #print(f'Starting scenario {scenario} with parameters {params_set} in run {run_number}')

    shareholder_pressure_lever, manager_pressure_lever, employee_pressure_lever, market_pressure_lever = params_set
    
    model = SBTiModel( num_companies= num_companies, 
        max_steps= number_of_steps,
                 
        # levers for experiments
        leadership_lever = leadership_lever, 
        risk_awareness_lever = risk_awareness_lever,
        reputation_lever = reputation_lever,
        shareholder_pressure_lever = shareholder_pressure_lever,
        manager_pressure_lever =manager_pressure_lever, 
        employee_pressure_lever= employee_pressure_lever,
        market_pressure_lever = market_pressure_lever,
        
        # network
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        delta = delta,
        rewiring_frequency = rewiring_frequency,
                 
        #awareness
        aware_update_step = aware_update_step,
        companies_turn_aware_per_round = companies_turn_aware_per_round,  # how often companies turn aware due to SBTi campaigns
        steps_per_round = steps_per_round,  
                 
        #Committing
        meetings_per_year = meetings_per_year,
        pres_mot_eval = pres_mot_eval, 
        
        manufacturing_coefficient =  manufacturing_coefficient,
        non_manufacturing_coefficient = non_manufacturing_coefficient,
        shareholder_pressure_coefficient = shareholder_pressure_coefficient,
        manager_pressure_coefficient = manager_pressure_coefficient,
        employee_pressure_coefficient = employee_pressure_coefficient,
        market_pressure_coefficient = market_pressure_coefficient,
        pressure_threshold = pressure_threshold,
        motivation_threshold = motivation_threshold,
        
                 
        #setting a target
        work_rate = work_rate,
        internal_target_range=internal_target_range,
        max_comm_duration = max_comm_duration,
        
        sigma=sigma, seed=seed)
    

    for _ in range(number_of_steps): # or however many steps you want
        model.step(shareholder_pressure_lever,
                   manager_pressure_lever,
                   employee_pressure_lever,
                   market_pressure_lever)
        
    data_model = model.datacollector.get_model_vars_dataframe()
    return scenario, params_set, run_number, data_model


if __name__ == "__main__":
    # Prepare a list of parameters for each run
    params = []
    for scenario, parameter_sets in scenarios.items():
        for i, params_set in enumerate(parameter_sets):
            for run_number in range(50):
                params.append((scenario, params_set, run_number))

    # Use a multiprocessing pool to run the simulations in parallel
    results = []
    with Pool() as pool:
        for i, result in enumerate(pool.imap_unordered(run_simulation, params), 1):
            results.append(result)
            print(f'Completed {i} out of {len(params)} runs')



    
    params = []
    for scenario, parameter_sets in scenarios.items():
        for params_set in parameter_sets:
            for run_number in range(50):
                print(run_number)
                params.append((scenario, params_set, run_number))
                
#%% Running the model to collect the data. Commented out

    #with Pool(os.cpu_count()) as pool:
     #   results = pool.map(run_simulation, params)


#%%

  #   # Prepare dictionaries to store the results
  #   all_model_results = {}
  # #  all_agent_results = {}

  #   # Unpack results and store them in dictionaries
  #   for result in results:
  #       scenario, params_set, run_number, data_model, data_agent = result
  #       parameter_index = scenarios[scenario].index(params_set)

  #       # Initialize dictionaries if necessary
  #       if scenario not in all_model_results:
  #           all_model_results[scenario] = {}
  #     #      all_agent_results[scenario] = {}
  #       if parameter_index not in all_model_results[scenario]:
  #           all_model_results[scenario][parameter_index] = []
  #         #  all_agent_results[scenario][parameter_index] = []

  #       # Store data
  #       all_model_results[scenario][parameter_index].append(data_model)
  #      # all_agent_results[scenario][parameter_index].append(data_agent)
    

    
#%%    
#    import pickle
    
#    with open('data_collected/all_model_results_2233.pickle', 'wb') as f:
#        pickle.dump(all_model_results, f)
    

#%%
    import pickle
    
    with open('data_collected/all_model_results_2233.pickle', 'rb') as f:
        all_model_results = pickle.load(f)
    

 
#%%    
    # Compute averages and standard deviation over the 50 runs for each scenario and each variable
    averaged_results = {}
    std_dev_results = {}
    
    for scenario, runs in all_model_results.items():
        averaged_results[scenario] = {}
        std_dev_results[scenario] = {}
        for i, dataframes in runs.items():
            # Compute the mean and standard deviation over the 50 dataframes for each variable
            averaged_results[scenario][i] = pd.concat(dataframes).groupby(level=0).mean()
            std_dev_results[scenario][i] = pd.concat(dataframes).groupby(level=0).std()

    
    #%%
    
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a colormap to get different colors for each scenario
    colormap = plt.cm.get_cmap('Set1', len(scenarios))
    
    # Plot lines for each scenario
    for scenario_num, scenario in enumerate(averaged_results.keys()):
        for i, params in enumerate(scenarios[scenario]):
            if 5 in params or scenario == 'NoIncrease':
                data_model = averaged_results[scenario][i]
    
                # Split data into before and after step 64
                data_model_before_64 = data_model.loc[data_model.index < 64]
                data_model_after_64 = data_model.loc[data_model.index >= 64]
    
                # Plot commitments
                ax.plot(data_model_before_64.index, data_model_before_64['Committed'], color='blue')
                ax.plot(data_model_after_64.index, data_model_after_64['Committed'], color=colormap(scenario_num), label=f'{scenario} - committed')
    
                # Plot set targets
                ax.plot(data_model_before_64.index, data_model_before_64['HasTarget'], color='red')
                ax.plot(data_model_after_64.index, data_model_after_64['HasTarget'], color=colormap(scenario_num), label=f'{scenario} - set targets')
    
    # Set plot title and labels
    ax.set_title('Commitments and Set Targets Over Time')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Number of Companies')
    
    # Show legend
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', borderaxespad=0., ncol=2)
    
    
    
    # Show plot
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    
    
    #%%
    # Define a color mapping for each scenario
    color_map = {
        'ShareholderPressure': 'blue',
        'ManagerPressure': 'green',
        'EmployeePressure': 'red',
        'MarketPressure': 'purple',
        'AllPressure': 'orange',
        'NoIncrease': 'black'
    }
    
    fig, axs = plt.subplots(2, figsize=(10, 15))  # Create two subplots
    
    # Parameter sets for each scenario
    param_sets = {
        'ShareholderPressure':  [(5,1,1,1), (10,1,1,1), (15,1,1,1)],
        'ManagerPressure':      [(1,5,1,1), (1,10,1,1), (1,15,1,1)],
        'EmployeePressure':     [(1,1,5,1), (1,1,10,1), (1,1,15,1)],
        'MarketPressure':       [(1,1,1,5), (1,1,1,10), (1,1,1,15)],
        'AllPressure':          [(5,5,5,5), (10,10,10,10), (15,15,15,15)],
        'NoIncrease':           [(1,1,1,1)]
    }
    
    
    # Define your desired parameter set for each scenario
    desired_params = {
        'ShareholderPressure':  (5,1,1,1),
        'ManagerPressure':      (1,5,1,1),
        'EmployeePressure':     (1,1,5,1),
        'MarketPressure':       (1,1,1,5),
        'AllPressure':          (5,5,5,5),
        'NoIncrease':           (1,1,1,1)
    }
    
    # Plot lines for each scenario with standard deviation as gray shade
    for scenario, _ in param_sets.items():
        color = color_map[scenario]  # Fetch the color for this scenario
        
        # Get the index of the desired parameters in the original parameter list
        params_index = param_sets[scenario].index(desired_params[scenario])
        
        # Committed subplot
        axs[0].plot(averaged_results[scenario][params_index].index[:64], 
                    averaged_results[scenario][params_index]['Committed'][:64], 
                    color='blue')
        axs[0].fill_between(averaged_results[scenario][params_index].index[:64],
                            (averaged_results[scenario][params_index]['Committed'] - std_dev_results[scenario][params_index]['Committed'])[:64],
                            (averaged_results[scenario][params_index]['Committed'] + std_dev_results[scenario][params_index]['Committed'])[:64],
                            color='gray', alpha=0.5)
        axs[0].plot(averaged_results[scenario][params_index].index[64:], 
                    averaged_results[scenario][params_index]['Committed'][64:], 
                    color=color, label=scenario)
        axs[0].fill_between(averaged_results[scenario][params_index].index[64:],
                            (averaged_results[scenario][params_index]['Committed'] - std_dev_results[scenario][params_index]['Committed'])[64:],
                            (averaged_results[scenario][params_index]['Committed'] + std_dev_results[scenario][params_index]['Committed'])[64:],
                            color='gray', alpha=0.5)
        
        # Set Targets subplot
        axs[1].plot(averaged_results[scenario][params_index].index[:64], 
                    averaged_results[scenario][params_index]['HasTarget'][:64], 
                    color='red')
        axs[1].fill_between(averaged_results[scenario][params_index].index[:64],
                            (averaged_results[scenario][params_index]['HasTarget'] - std_dev_results[scenario][params_index]['HasTarget'])[:64],
                            (averaged_results[scenario][params_index]['HasTarget'] + std_dev_results[scenario][params_index]['HasTarget'])[:64],
                            color='gray', alpha=0.5)
        axs[1].plot(averaged_results[scenario][params_index].index[64:], 
                    averaged_results[scenario][params_index]['HasTarget'][64:], 
                    color=color, label=scenario)
        axs[1].fill_between(averaged_results[scenario][params_index].index[64:],
                            (averaged_results[scenario][params_index]['HasTarget'] - std_dev_results[scenario][params_index]['HasTarget'])[64:],
                            (averaged_results[scenario][params_index]['HasTarget'] + std_dev_results[scenario][params_index]['HasTarget'])[64:],
                            color='gray', alpha=0.5)    
    # Set plot titles and labels
    axs[0].set_title('Commitments Over Time')
    axs[1].set_title('Set Targets Over Time')
    for ax in axs:
        ax.set_xlabel('Time step')
        ax.set_ylabel('Number of Companies')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    



#%%
    total_companies = 2233
    
    fig, axs = plt.subplots(2, figsize=(10, 15))  # Create two subplots
    
    # Plot lines for each scenario
    for scenario, _ in param_sets.items():
        color = color_map[scenario]  # Fetch the color for this scenario
    
        # Get the index of the desired parameters in the original parameter list
        params_index = param_sets[scenario].index(desired_params[scenario])
    
        # Committed subplot
        averaged_values = averaged_results[scenario][params_index]['Committed'] * 100 / total_companies
        std_values = std_dev_results[scenario][params_index]['Committed'] * 100 / total_companies
        time_index = averaged_results[scenario][params_index].index
        axs[0].plot(time_index[:64], averaged_values[:64], color='blue')
        axs[0].plot(time_index[64:], averaged_values[64:], color=color, label=scenario)
        axs[0].fill_between(time_index, (averaged_values-std_values), (averaged_values+std_values), color=color, alpha=.1)
        axs[0].axhline(y=20, color='grey', linestyle='--')  # Add horizontal line at 20%
        axs[0].axvline(x=60, color='grey', linestyle='--')  # Add vertical line at step 60

        # Set Targets subplot
        averaged_values = averaged_results[scenario][params_index]['HasTarget'] * 100 / total_companies
        std_values = std_dev_results[scenario][params_index]['HasTarget'] * 100 / total_companies
        axs[1].plot(time_index[:64], averaged_values[:64], color='red')
        axs[1].plot(time_index[64:], averaged_values[64:], color=color, label=scenario)
        axs[1].fill_between(time_index, (averaged_values-std_values), (averaged_values+std_values), color=color, alpha=.1)
        axs[1].axhline(y=12, color='grey', linestyle='--')  # Add horizontal line at 12%
        axs[1].axvline(x=60, color='grey', linestyle='--')  # Add vertical line at step 60

    # Set plot titles and labels
    axs[0].set_title('Commitments Over Time')
    axs[1].set_title('Set Targets Over Time')
    for ax in axs:
        ax.set_xlabel('Time step')
        ax.set_ylabel('Percentage of Companies (%)')
        ax.legend()
    
    plt.tight_layout()
    plt.show()


    #%% convert the step numbers to dates starting from September 2015- Εach step is a month
    from matplotlib.dates import date2num, num2date
    desired_params = {
        'ShareholderPressure':  (5,1,1,1),
        'ManagerPressure':      (1,5,1,1),
        'EmployeePressure':     (1,1,5,1),
        'MarketPressure':       (1,1,1,5),
        'AllPressure':          (5,5,5,5),
        'NoIncrease':           (1,1,1,1)
    }
    start_date = pd.to_datetime('2015-09-01')
    
    # Plot for 'Commitments Over Time'
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    for scenario, _ in param_sets.items():
        color = color_map[scenario]  # Fetch the color for this scenario
        params_index = param_sets[scenario].index(desired_params[scenario])
    
        dates = pd.date_range(start=start_date, periods=len(averaged_results[scenario][params_index].index), freq='M')
    
        # Committed plot
        ax1.plot(dates[:64], 
                averaged_results[scenario][params_index]['Committed'][:64]/total_companies * 100, 
                color='green')
        ax1.plot(dates[63:], 
                averaged_results[scenario][params_index]['Committed'][63:]/total_companies * 100, 
                color=color, label=scenario)
        ax1.fill_between(dates, 
                        (averaged_results[scenario][params_index]['Committed'] - std_dev_results[scenario][params_index]['Committed'])/total_companies * 100,
                        (averaged_results[scenario][params_index]['Committed'] + std_dev_results[scenario][params_index]['Committed'])/total_companies * 100, 
                        color=color, alpha=0.1)
    
  #  ax1.axhline(y=20, color='grey', linestyle='--')  # Add horizontal line at 20%
    ax1.axvline(x=date2num(start_date + pd.DateOffset(months=64)), color='grey', linestyle='--')  # Add vertical line at step 60
    #ax1.set_title('Commitments Over Time', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Percentage of Companies (%)', fontsize=14)
    ax1.set_ylim(0, 100)  # set y-axis limits
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    #plt.savefig('report/experiments_multiplier5_commitments.png')  # replace with your desired file path and name
    plt.show()
    
    # Plot for 'Set Targets Over Time'
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    for scenario, _ in param_sets.items():
        color = color_map[scenario]  # Fetch the color for this scenario
        params_index = param_sets[scenario].index(desired_params[scenario])
    
        # Set Targets plot
        ax2.plot(dates[:64], 
                averaged_results[scenario][params_index]['HasTarget'][:64]/total_companies * 100, 
                color='red')
        ax2.plot(dates[63:], 
                averaged_results[scenario][params_index]['HasTarget'][63:]/total_companies * 100, 
                color=color, label=scenario)
        ax2.fill_between(dates, 
                        (averaged_results[scenario][params_index]['HasTarget'] - std_dev_results[scenario][params_index]['HasTarget'])/total_companies * 100,
                        (averaged_results[scenario][params_index]['HasTarget'] + std_dev_results[scenario][params_index]['HasTarget'])/total_companies * 100, 
                        color=color, alpha=0.1)
    
 #   ax2.axhline(y=12, color='grey', linestyle='--')  # Add horizontal line at 12%
    ax2.axvline(x=date2num(start_date + pd.DateOffset(months=64)), color='grey', linestyle='--')  # Add vertical line at step 60
    #ax2.set_title('Set Targets Over Time', fontsize=14)
    ax2.set_xlabel('Date', fontsize = 14)
    ax2.set_ylabel('Percentage of Companies (%)',fontsize=14)
    ax2.set_ylim(0, 100)  # set y-axis limits
    ax2.legend(fontsize=12)

    ax2.tick_params(axis='both', which='major', labelsize=12)  # Adjust the size of the ticks (numbers on the axes)

    
    plt.tight_layout()
    #plt.savefig('report/experiments_multiplier5_targets.png', dpi= 150)  # replace with your desired file path and name
    plt.show()
    
    differences = {
        scenario: {
            "CommitmentIncrease": averaged_results[scenario]['Committed'].iloc[-1] - averaged_results[scenario]['Committed'].iloc[63],
            "SetTargetIncrease": averaged_results[scenario]['HasTarget'].iloc[-1] - averaged_results[scenario]['HasTarget'].iloc[63]
        }
        for scenario in color_map.keys()
    }
    
    df_differences = pd.DataFrame(differences).T
    print(df_differences)
    
    
#%%
    # Define your desired parameter set for each scenario
    desired_params = {
        'ShareholderPressure':  (10,1,1,1),
        'ManagerPressure':      (1,10,1,1),
        'EmployeePressure':     (1,1,10,1),
        'MarketPressure':       (1,1,1,10),
        'AllPressure':          (10,10,10,10),
        'NoIncrease':           (1,1,1,1)
    }
    #%% convert the step numbers to dates starting from September 2015- Εach step is a month
    from matplotlib.dates import date2num, num2date
    
    start_date = pd.to_datetime('2015-09-01')
    
    # Plot for 'Commitments Over Time'
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    for scenario, _ in param_sets.items():
        color = color_map[scenario]  # Fetch the color for this scenario
        params_index = param_sets[scenario].index(desired_params[scenario])
    
        dates = pd.date_range(start=start_date, periods=len(averaged_results[scenario][params_index].index), freq='M')
    
        # Committed plot
        ax1.plot(dates[:64], 
                averaged_results[scenario][params_index]['Committed'][:64]/total_companies * 100, 
                color='green')
        ax1.plot(dates[63:], 
                averaged_results[scenario][params_index]['Committed'][63:]/total_companies * 100, 
                color=color, label=scenario)
        ax1.fill_between(dates, 
                        (averaged_results[scenario][params_index]['Committed'] - std_dev_results[scenario][params_index]['Committed'])/total_companies * 100,
                        (averaged_results[scenario][params_index]['Committed'] + std_dev_results[scenario][params_index]['Committed'])/total_companies * 100, 
                        color=color, alpha=0.1)
    
    #ax1.axhline(y=20, color='grey', linestyle='--')  # Add horizontal line at 20%
    ax1.axvline(x=date2num(start_date + pd.DateOffset(months=64)), color='grey', linestyle='--')  # Add vertical line at step 60
    #ax1.set_title('Commitments Over Time')
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Percentage of Companies (%)',fontsize=14)
    ax1.set_ylim(0, 100)  # set y-axis limits
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    #plt.savefig('report_images/experiments_multiplier10_commitments.png')  # replace with your desired file path and name
    plt.show()
    
    # Plot for 'Set Targets Over Time'
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    for scenario, _ in param_sets.items():
        color = color_map[scenario]  # Fetch the color for this scenario
        params_index = param_sets[scenario].index(desired_params[scenario])
    
        # Set Targets plot
        ax2.plot(dates[:64], 
                averaged_results[scenario][params_index]['HasTarget'][:64]/total_companies * 100, 
                color='red')
        ax2.plot(dates[63:], 
                averaged_results[scenario][params_index]['HasTarget'][63:]/total_companies * 100, 
                color=color, label=scenario)
        ax2.fill_between(dates, 
                        (averaged_results[scenario][params_index]['HasTarget'] - std_dev_results[scenario][params_index]['HasTarget'])/total_companies * 100,
                        (averaged_results[scenario][params_index]['HasTarget'] + std_dev_results[scenario][params_index]['HasTarget'])/total_companies * 100, 
                        color=color, alpha=0.1)
    
    #ax2.axhline(y=12, color='grey', linestyle='--')  # Add horizontal line at 12%
    ax2.axvline(x=date2num(start_date + pd.DateOffset(months=64)), color='grey', linestyle='--')  # Add vertical line at step 60
    #ax2.set_title('Set Targets Over Time')
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Percentage of Companies (%)', fontsize=14)
    ax2.set_ylim(0, 100)  # set y-axis limits
    ax2.legend(fontsize=12)  
    
    ax2.tick_params(axis='both', which='major', labelsize=12)  # Adjust the size of the ticks (numbers on the axes)

    
    plt.tight_layout()
    #plt.savefig('report_images/experiments_multiplier10_targets.png')  # replace with your desired file path and name
    plt.show()
    