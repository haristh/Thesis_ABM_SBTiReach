#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:38:53 2023

@author: charistheodorou

One of the three python files that focus on the experimentation of the thesis. 
This one presents the second set of scenarios i.e. the scenarios that affect the model's 
mechanisms. The model runs from 2015 until 2025 (108 steps) with 2233 companies (total) 50 times and averages and stds 
are calculated.  Multiprocessing was used to reduce the time taken.

Running the data collection, saving figures and saving any data is commented out to avoid overwritin



"""

import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SBTiModel_scenarios_set2 import SBTiModel 
import os
from multiprocessing import Pool
import pickle
import time
import datetime
#%%
start_time = time.time()


num_companies = 2233
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


manufacturing_coefficient = 0.232
non_manufacturing_coefficient = 0.358
shareholder_pressure_coefficient = 0.253
manager_pressure_coefficient = 0.254
employee_pressure_coefficient = 0.238
market_pressure_coefficient = 0.210
 
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
    'BaseScenario': [(1, False, 10, 24)],  # No change to connectivity lever and no market pressure campaign
#    'MarketPressureCampaign': [(13, True, 10, 24)],  # Increase connectivity lever after step 64, activate market pressure
#    'ManagerPressure': [(1, False, 5, 24),],
    'LongerDeadline':[(1, False, 10, 48)],
#    'ShorterDeadline':[(1, False, 10, 12)],
    
}
#%%
all_model_results = {}
#all_agent_results = {}


def run_simulation(params):
    scenario, params_set, run_number = params

    # Unpacking parameters
    connectivity_lever, market_pressure_on, meetings_per_year_scenario, max_comm_duration_scenario = params_set  
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
        
        connectivity_lever = connectivity_lever,
        market_pressure_on = market_pressure_on,
        meetings_per_year_scenario = meetings_per_year_scenario,         
        max_comm_duration_scenario = max_comm_duration_scenario,
        
        
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
                   market_pressure_lever,
                   connectivity_lever, meetings_per_year_scenario, max_comm_duration_scenario)
        
    data_model = model.datacollector.get_model_vars_dataframe()
    #data_agent = model.datacollector.get_agent_vars_dataframe()
    return scenario, params_set, run_number, data_model #,data_agent


if __name__ == "__main__":
    # Prepare a list of parameters for each run
    params = []
    for scenario, parameter_sets in scenarios.items():
        for i, params_set in enumerate(parameter_sets):
            for run_number in range(50):
                params.append((scenario, params_set, run_number))


#%% Running the model to collect the data. Commented out
    # Use a multiprocessing pool to run the simulations in parallel
    #results = []
    # with Pool() as pool:
    #     for i, result in enumerate(pool.imap_unordered(run_simulation, params), 1):
    #         results.append(result)
    #         print(f'Completed {i} out of {len(params)} runs')

#%% save

    # final_data is the object you want to save
    #with open('data_collected/results_secondsetsenarios_2233.pickle', 'wb') as f:
    #     pickle.dump(results, f)           

#%% load
    with open('data_collected/results_secondsetsenarios_2233.pickle', 'rb') as f:
        results = pickle.load(f)

#%% Progress over time
    # Convert results into a dictionary format
    all_model_results = {}
    for scenario, params_set, run_number,data_model in results:
        if scenario not in all_model_results:
            all_model_results[scenario] = []
        all_model_results[scenario].append(data_model)
        # Compute averages and standard deviation over the 50 runs for each scenario and each variable
        averaged_results = {}
        std_dev_results = {}
    
    for scenario, runs in all_model_results.items():
        averaged_results[scenario] = pd.concat([run for run in runs]).groupby(level=0).mean()
        std_dev_results[scenario] = pd.concat([run for run in runs]).groupby(level=0).std()
    
    # Define a color mapping for each scenario
    color_map = {
        'BaseScenario': 'black',
        'MarketPressureCampaign': 'green',
        'ManagerPressure': 'pink',
        'LongerDeadline':'purple',
        'ShorterDeadline':'orange',
    }
    
    fig, axs = plt.subplots(2, figsize=(10, 15))  # Create two subplots
    
    # Plot lines for each scenario with standard deviation as gray shade
    for scenario in color_map.keys():
        color = color_map[scenario]  # Fetch the color for this scenario
    
        # Committed subplot
        axs[0].plot(averaged_results[scenario].index[:64],
                    averaged_results[scenario]['Committed'][:64],
                    color='blue')
        axs[0].fill_between(averaged_results[scenario].index[:64],
                            (averaged_results[scenario]['Committed'] - std_dev_results[scenario]['Committed'])[:64],
                            (averaged_results[scenario]['Committed'] + std_dev_results[scenario]['Committed'])[:64],
                            color='gray', alpha=0.5)
        axs[0].plot(averaged_results[scenario].index[63:],
                    averaged_results[scenario]['Committed'][63:],
                    color=color, label=scenario)
        axs[0].fill_between(averaged_results[scenario].index[63:],
                            (averaged_results[scenario]['Committed'] - std_dev_results[scenario]['Committed'])[63:],
                            (averaged_results[scenario]['Committed'] + std_dev_results[scenario]['Committed'])[63:],
                            color='gray', alpha=0.5)
    
        # Set Targets subplot
        axs[1].plot(averaged_results[scenario].index[:64],
                    averaged_results[scenario]['HasTarget'][:64],
                    color='red')
        axs[1].fill_between(averaged_results[scenario].index[:64],
                            (averaged_results[scenario]['HasTarget'] - std_dev_results[scenario]['HasTarget'])[:64],
                            (averaged_results[scenario]['HasTarget'] + std_dev_results[scenario]['HasTarget'])[:64],
                            color='gray', alpha=0.5)
        axs[1].plot(averaged_results[scenario].index[63:],
                    averaged_results[scenario]['HasTarget'][63:],
                    color=color, label=scenario)
        axs[1].fill_between(averaged_results[scenario].index[63:],
                            (averaged_results[scenario]['HasTarget'] - std_dev_results[scenario]['HasTarget'])[63:],
                            (averaged_results[scenario]['HasTarget'] + std_dev_results[scenario]['HasTarget'])[63:],
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

    # Extract the evolution data from each run
    aware_data = [result[3]["Aware"] for result in results]
    committed_data = [result[3]["Committed"] for result in results]
    set_target_data = [result[3]["HasTarget"] for result in results]
    
    # Convert lists of series to DataFrames
    df_aware = pd.concat(aware_data, axis=1)
    df_committed = pd.concat(committed_data, axis=1)
    df_set_target = pd.concat(set_target_data, axis=1)
    
    # Convert the step numbers to dates
    start_date = pd.to_datetime('2015-05-01')
    date_index = pd.date_range(start=start_date, periods=number_of_steps, freq='M')
    
    # Calculate the mean values and standard errors, convert to percentages
    mean_aware = (df_aware.mean(axis=1) / num_companies) * 100
    mean_committed = (df_committed.mean(axis=1) / num_companies) * 100
    mean_set_target = (df_set_target.mean(axis=1) / num_companies) * 100
    
    stderr_aware = (df_aware.std(axis=1) / np.sqrt(len(results))) / num_companies * 100
    stderr_committed = (df_committed.std(axis=1) / np.sqrt(len(results))) / num_companies * 100
    stderr_set_target = (df_set_target.std(axis=1) / np.sqrt(len(results))) / num_companies * 100
    
    # Plot the means over time with standard error shading
    plt.figure(figsize=(10, 6))
    plt.plot(date_index, mean_aware, label='Aware', color='blue')
    plt.fill_between(date_index, mean_aware - stderr_aware, mean_aware + stderr_aware, color='blue', alpha=0.2)
    plt.plot(date_index, mean_committed, label='Committed', color='green')
    plt.fill_between(date_index, mean_committed - stderr_committed, mean_committed + stderr_committed, color='green', alpha=0.2)
    plt.plot(date_index, mean_set_target, label='HasTarget', color='red')
    plt.fill_between(date_index, mean_set_target - stderr_set_target, mean_set_target + stderr_set_target, color='red', alpha=0.2)
    plt.legend(loc='best')
    #plt.title('Evolution of company states over time (mean over runs)')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Percentage of companies (%)', fontsize=14)
    plt.grid(True)
    plt.xlim([datetime.date(2015, 6, 1), datetime.date(2020, 5, 1)])
    #plt.savefig("images_experiments/base_scenario_2233_progress.png")
 
    #plt.savefig("report_images/progression.png", dpi = 500, bbox_inches='tight')
    plt.show()

    
    
    plt.show()

#%%average time

    # Extract the "AverageCommitToTargetTime" from the final step of each run
    average_commit_to_target_time_data = [result[3]["AverageCommitToTargetTime"].iloc[-1] for result in results]

    # Calculate the mean value across all runs
    average_commit_to_target_time_mean = np.mean(average_commit_to_target_time_data)
    
    # Calculate the standard deviation
    average_commit_to_target_time_std = np.std(average_commit_to_target_time_data)
    
    # Calculate the standard error
    average_commit_to_target_time_se = average_commit_to_target_time_std / np.sqrt(len(average_commit_to_target_time_data))
    
    print(f"Average of 'AverageCommitToTargetTime' in the last step across all runs: {average_commit_to_target_time_mean} ± {average_commit_to_target_time_se}")

#%% Final results

    # Extract the data for the last step
    last_step_aware = [result[3]["Aware"].iloc[-1] for result in results]
    last_step_committed = [result[3]["Committed"].iloc[-1] for result in results]
    last_step_set_target = [result[3]["HasTarget"].iloc[-1] for result in results]
    
    # Calculate means (as percentages)
    last_step_aware_mean = np.mean(last_step_aware) * 100 / num_companies
    last_step_committed_mean = np.mean(last_step_committed) * 100 / num_companies
    last_step_set_target_mean = np.mean(last_step_set_target) * 100 / num_companies
    
    # Calculate stds
    last_step_aware_std = np.std(last_step_aware) * 100 / num_companies
    last_step_committed_std = np.std(last_step_committed) * 100 / num_companies
    last_step_set_target_std = np.std(last_step_set_target) * 100 / num_companies
    
    # Calculate standard errors
    last_step_aware_se = last_step_aware_std / np.sqrt(len(last_step_aware))
    last_step_committed_se = last_step_committed_std / np.sqrt(len(last_step_committed))
    last_step_set_target_se = last_step_set_target_std / np.sqrt(len(last_step_set_target))
    
    print(f"Percentage of 'Aware' companies in the last step across all runs: {last_step_aware_mean:.2f}% ± {last_step_aware_se:.2f}%")
    print(f"Percentage of 'Committed' companies in the last step across all runs: {last_step_committed_mean:.2f}% ± {last_step_committed_se:.2f}%")
    print(f"Percentage of 'SetTarget' companies in the last step across all runs: {last_step_set_target_mean:.2f}% ± {last_step_set_target_se:.2f}%")
    


#%%     Sectors 108 steps

    # Filter to only include 'BaseScenario' results
    results_copy = [result for result in results if result[0] == 'BaseScenario']

    model_data = [result[3] for result in results_copy]  # result[3] is the data_model
    
    # Aggregate the collected model data
    aggregated_data = pd.concat(model_data, keys=range(len(model_data)))
    # Group the data by the first level of the index (which is the run number) and get the last entry of each group
    last_step_data = aggregated_data.groupby(level=0).last()
    
    # # Now, iterate over the rows of this new DataFrame
    # for idx, row in last_step_data.iterrows():
    #     print(row["TotalBySector"] == row["AwareBySector"])


    # Convert the dictionary columns into a multi-index DataFrame
    aggregated_data_expanded = pd.concat(
        [aggregated_data[col].apply(pd.Series) for col in ["TotalBySector", "AwareBySector", "CommittedBySector", "SetTargetBySector"]], 
        keys=["Total", "Aware", "Committed", "SetTarget"], 
        axis=1
    )
    
    # Extract the data for the last step
    last_step_total = aggregated_data_expanded["Total"].groupby(level=0).last()
    last_step_aware = aggregated_data_expanded["Aware"].groupby(level=0).last()
    last_step_committed = aggregated_data_expanded["Committed"].groupby(level=0).last()
    last_step_set_target = aggregated_data_expanded["SetTarget"].groupby(level=0).last()
    
    # Calculate means
    last_step_total_mean = last_step_total.mean()
    last_step_aware_mean = last_step_aware.mean()
    last_step_committed_mean = last_step_committed.mean()
    last_step_set_target_mean = last_step_set_target.mean()
    
    # Calculate stds
    last_step_total_std = last_step_total.std()
    last_step_aware_std = last_step_aware.std()
    last_step_committed_std = last_step_committed.std()
    last_step_set_target_std = last_step_set_target.std()
    
    # Calculate standard errors
    last_step_total_se = last_step_total_std / np.sqrt(len(last_step_total))
    last_step_aware_se = last_step_aware_std / np.sqrt(len(last_step_aware))
    last_step_committed_se = last_step_committed_std / np.sqrt(len(last_step_committed))
    last_step_set_target_se = last_step_set_target_std / np.sqrt(len(last_step_set_target))
    
    # Create a DataFrame from the mean series
    df_sector_mean = pd.DataFrame({'Total': last_step_total_mean, 'Aware': last_step_aware_mean, 'Committed': last_step_committed_mean, 'Set Target': last_step_set_target_mean})
    
    # Create a DataFrame from the std series
    df_sector_std = pd.DataFrame({'Total': last_step_total_std, 'Aware': last_step_aware_std, 'Committed': last_step_committed_std, 'Set Target': last_step_set_target_std})
    
    # Create a DataFrame from the se series
    df_sector_se = pd.DataFrame({'Total': last_step_total_se, 'Aware': last_step_aware_se, 'Committed': last_step_committed_se, 'Set Target': last_step_set_target_se})
    
    # Create a bar plot with error bars
    ax = df_sector_mean.plot(kind="bar", yerr=df_sector_std, figsize=(10, 7), capsize=4)
    #plt.title("Average Number of Companies in Each State per Sector with SD")
    plt.ylabel("Number of Companies")
    plt.xlabel("Sector")
    plt.ylim(bottom=0)  # Set lower limit to 0
    #plt.savefig("report_images/images_experiments/base_scenario_2233_sectors.png")
    plt.show()

    # Calculate percentages
    last_step_committed_percentage = last_step_committed_mean / last_step_total_mean * 100
    last_step_set_target_percentage = last_step_set_target_mean / last_step_total_mean * 100
    
    # Create a DataFrame from the percentage series
    df_sector_percentage = pd.DataFrame({'Committed (%)': last_step_committed_percentage, 'Set Target (%)': last_step_set_target_percentage})

    # Round the values to one decimal place
    df_sector_percentage = df_sector_percentage.round(1)
    
    # Sort the DataFrame by the 'Committed (%)' column in descending order
    df_sector_percentage = df_sector_percentage.sort_values('Committed (%)', ascending=False)
    
    # Print the DataFrame
    print(df_sector_percentage)
    
    
    

#%%     Sectors 60 steps

    # Filter to only include 'BaseScenario' results
    results_copy = [result for result in results if result[0] == 'BaseScenario']

    model_data = [result[3] for result in results_copy]  # result[3] is the data_model
    
    # Aggregate the collected model data
    aggregated_data = pd.concat(model_data, keys=range(len(model_data)))

    # Group the data by the first level of the index (which is the run number) and get the entry for step 60 of each group
    step_60_data = aggregated_data.xs(60, level=1)
    

    # Extract the data for step 60
    step_60_total = step_60_data["TotalBySector"].apply(pd.Series)
    step_60_aware = step_60_data["AwareBySector"].apply(pd.Series)
    step_60_committed = step_60_data["CommittedBySector"].apply(pd.Series)
    step_60_set_target = step_60_data["SetTargetBySector"].apply(pd.Series)
    
    # Calculate means
    step_60_total_mean = step_60_total.mean()
    step_60_aware_mean = step_60_aware.mean()
    step_60_committed_mean = step_60_committed.mean()
    step_60_set_target_mean = step_60_set_target.mean()
    
    # Calculate stds
    step_60_total_std = step_60_total.std()
    step_60_aware_std = step_60_aware.std()
    step_60_committed_std = step_60_committed.std()
    step_60_set_target_std = step_60_set_target.std()
    
    # Calculate standard errors
    step_60_total_se = step_60_total_std / np.sqrt(len(step_60_total))
    step_60_aware_se = step_60_aware_std / np.sqrt(len(step_60_aware))
    step_60_committed_se = step_60_committed_std / np.sqrt(len(step_60_committed))
    step_60_set_target_se = step_60_set_target_std / np.sqrt(len(step_60_set_target))
    
    # Create a DataFrame from the mean series
    df_sector_mean = pd.DataFrame({'Total': step_60_total_mean, 'Committed': step_60_committed_mean, 'Set Target': step_60_set_target_mean})
    
    # Create a DataFrame from the std series
    df_sector_std = pd.DataFrame({'Total': step_60_total_std, 'Committed': step_60_committed_std, 'Set Target': step_60_set_target_std})
    
    # Create a DataFrame from the se series
    df_sector_se = pd.DataFrame({'Total': step_60_total_se, 'Committed': step_60_committed_se, 'Set Target': step_60_set_target_se})
    
    # Create a bar plot with error bars
    ax = df_sector_mean.plot(kind="bar", yerr=df_sector_std, figsize=(14, 10), capsize=4, color=['skyblue', 'mediumseagreen', 'lightcoral'])
    
    plt.ylabel("Number of Companies", fontsize=14)
    plt.xlabel("Sector", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)  # Set lower limit to 0
    #plt.savefig("report_images/base_scenario_sectors.png", dpi = 500, bbox_inches='tight')
    plt.show()

    
    
    # Calculate percentages
    step_60_committed_percentage = step_60_committed_mean / step_60_total_mean * 100
    step_60_set_target_percentage = step_60_set_target_mean / step_60_total_mean * 100
    
    # Create a DataFrame from the percentage series
    df_sector_percentage = pd.DataFrame({'Committed (%)': step_60_committed_percentage, 'Set Target (%)': step_60_set_target_percentage})
    
    print(df_sector_mean)
    # Save dataframe to a csv file
    #df_sector_mean.to_csv('data_collected/sector_mean.csv', index='Sector')
    
    # Round the values to one decimal place
    df_sector_percentage = df_sector_percentage.round(1)
    
    # Sort the DataFrame by the 'Committed (%)' column in descending order
    df_sector_percentage = df_sector_percentage.sort_values('Committed (%)', ascending=False)
    
    
    
    
    
    #%% Emissions step 60

    # Assuming step_60_aware_mean, step_60_committed_mean, and step_60_set_target_mean have been calculated previously
    # for the selected scenario and step 60.
    

    import seaborn as sns
    
    # Read sector data from Excel
    sector_df = pd.read_excel("files/StructuredData.xlsx", sheet_name='SectorsUsed', header=0,
                              index_col='Sector', nrows=8)
    
    # Calculate total emissions
    total_emissions_committed = (step_60_committed_mean * sector_df['Emissions per company (Mt)']).sum()
    total_emissions_target = (step_60_set_target_mean * sector_df['Emissions per company (Mt)']).sum()
    total_emissions_aware = (step_60_aware_mean * sector_df['Emissions per company (Mt)']).sum()
    
    # Calculate emissions per company
    emissions_per_company_committed = total_emissions_committed / step_60_committed_mean.sum()
    emissions_per_company_target = total_emissions_target / step_60_set_target_mean.sum()
    emissions_per_company_aware = total_emissions_aware / step_60_aware_mean.sum()
    
    # Create a DataFrame with total emissions and emissions per company
    df_total_emissions = pd.DataFrame({
        'Total': [total_emissions_aware, emissions_per_company_aware],
        'Committed': [total_emissions_committed, emissions_per_company_committed], 
        'Set Target': [total_emissions_target, emissions_per_company_target],

    }, index=['Total Emissions (Mt)', 'Emissions per Company (Mt)'])
    
    # Print the DataFrame for absolute emissions and emissions per company
    print("Emissions table:")
    print(df_total_emissions)
    
    # Filter the DataFrame to include only 'Committed' and 'Set Target' for the bar plot
    df_emissions_per_company = df_total_emissions.drop(columns='Total').loc[['Emissions per Company (Mt)']].reset_index()
    
    # Convert the DataFrame from wide to long format
    df_emissions_per_company_melt = df_emissions_per_company.melt(id_vars='index', var_name='State', value_name='Emissions (Mt)')
    
    # Use seaborn to plot the bars
    plt.figure(figsize=(10, 7))
    sns.barplot(x='State', y='Emissions (Mt)', hue='index', data=df_emissions_per_company_melt, palette=['mediumseagreen', 'lightcoral'])
    plt.ylabel("Emissions (Mt)")
    plt.xlabel("State")
    plt.show()


#%% Manufacturing or not step 60

    # Load the sector information
    sector_df = pd.read_excel("files/StructuredData.xlsx", sheet_name='SectorsUsed', header=0,
                              index_col='Sector', nrows = 8)
    
    # Prepare a dictionary mapping sectors to their types
    sector_to_type = sector_df["Sector Type"].to_dict()
    
    # Create a DataFrame from the mean series
    df_sector_mean = pd.DataFrame({'Total': step_60_total_mean, 'Aware': step_60_aware_mean,'Committed': step_60_committed_mean, 'Set Target': step_60_set_target_mean})
    
    # Create a DataFrame from the std series
    df_sector_std = pd.DataFrame({'Total': step_60_total_std, 'Aware': step_60_aware_std, 'Committed': step_60_committed_std, 'Set Target': step_60_set_target_std})
    
    # Create a DataFrame from the se series
    df_sector_se = pd.DataFrame({'Total': step_60_total_se,'Aware': step_60_aware_se, 'Committed': step_60_committed_se, 'Set Target': step_60_set_target_se})
    
    
    
    # Aggregate the data according to "Sector Type"
    df_sector_mean['Sector Type'] = df_sector_mean.index.map(sector_to_type)
    df_sector_std['Sector Type'] = df_sector_std.index.map(sector_to_type)
    df_sector_se['Sector Type'] = df_sector_se.index.map(sector_to_type)
    
    aggregated_mean = df_sector_mean.groupby('Sector Type').sum()
    aggregated_std = df_sector_std.groupby('Sector Type').sum()
    aggregated_se = df_sector_se.groupby('Sector Type').sum()
    
    # Plot the data
    ax = aggregated_mean.plot(kind="bar", yerr=aggregated_std, figsize=(10, 7), capsize=4)
    #plt.title("Average Number of Companies in Each State per Sector Type with SE")
    plt.ylabel("Number of Companies")
    plt.xlabel("Sector Type")
    plt.ylim(bottom=0)  # Set lower limit to 0
    #plt.savefig("images_experiments/base_scenario_2233_sectors_by_type.png")
    plt.show()
    
    # Calculate the percentages
    aggregated_mean_percentage = (aggregated_mean.iloc[:, 1:] / aggregated_mean['Total'].values.reshape(-1,1)) * 100
    
    
    
    # Add a 'Total' column in aggregated_mean_percentage filled with 100
    aggregated_mean_percentage['Total'] = 100

    # Reorder the columns so that 'Total' comes first
    aggregated_mean_percentage = aggregated_mean_percentage[['Total', 'Aware', 'Committed', 'Set Target']]
    
    # Create a new DataFrame for the output
    output_df = pd.concat([aggregated_mean, aggregated_mean_percentage], axis=1)
    
    # Rename the columns
    output_df.columns = pd.MultiIndex.from_product([['Number', 'Percentage'], ['Total', 'Aware', 'Committed', 'Set Target']])
    
    # Output the data
    print(output_df)


#%% Emissions

    sector_df = pd.read_excel("files/StructuredData.xlsx", sheet_name='SectorsUsed', header=0,
                              index_col='Sector', nrows = 8)
    
    # Calculate total emissions
    total_emissions_aware = (last_step_aware_mean * sector_df['Emissions per company (Mt)']).sum()
    total_emissions_committed = (last_step_committed_mean * sector_df['Emissions per company (Mt)']).sum()
    total_emissions_target = (last_step_set_target_mean * sector_df['Emissions per company (Mt)']).sum()

    # Create a DataFrame with total emissions
    df_total_emissions = pd.DataFrame({
        'Aware': [total_emissions_aware], 
        'Committed': [total_emissions_committed], 
        'Set Target': [total_emissions_target]
    })
    
    # Convert the DataFrame from wide to long format
    df_total_emissions_melt = df_total_emissions.melt(var_name='State', value_name='Total Emissions')
    
    # Use seaborn to plot the bars
    import seaborn as sns
    
    plt.figure(figsize=(10, 7))
    sns.barplot(x='State', y='Total Emissions', data=df_total_emissions_melt)
    #plt.title("Total Emissions by State of Companies per Sector")
    plt.ylabel("Total Emissions (Mt)")
    plt.xlabel("State")
    #plt.savefig("images_experiments/base_scenario_2233_emissions.png")
    plt.show()




#%% Manufacturing or not 

    # Load the sector information
    sector_df = pd.read_excel("files/StructuredData.xlsx", sheet_name='SectorsUsed', header=0,
                              index_col='Sector', nrows = 8)
    
    # Prepare a dictionary mapping sectors to their types
    sector_to_type = sector_df["Sector Type"].to_dict()
    
    # Aggregate the data according to "Sector Type"
    df_sector_mean['Sector Type'] = df_sector_mean.index.map(sector_to_type)
    df_sector_std['Sector Type'] = df_sector_std.index.map(sector_to_type)
    df_sector_se['Sector Type'] = df_sector_se.index.map(sector_to_type)
    
    aggregated_mean = df_sector_mean.groupby('Sector Type').sum()
    aggregated_std = df_sector_std.groupby('Sector Type').sum()
    aggregated_se = df_sector_se.groupby('Sector Type').sum()
    
    # Plot the data
    ax = aggregated_mean.plot(kind="bar", yerr=aggregated_std, figsize=(10, 7), capsize=4)
    #plt.title("Average Number of Companies in Each State per Sector Type with SE")
    plt.ylabel("Number of Companies")
    plt.xlabel("Sector Type")
    plt.ylim(bottom=0)  # Set lower limit to 0
    #plt.savefig("images_experiments/base_scenario_2233_sectors_by_type.png")
    plt.show()
    
    # Calculate the percentages
    aggregated_mean_percentage = (aggregated_mean.iloc[:, 1:] / aggregated_mean['Total'].values.reshape(-1,1)) * 100
    
    

    
    
    # Add a 'Total' column in aggregated_mean_percentage filled with 100
    aggregated_mean_percentage['Total'] = 100

    # Reorder the columns so that 'Total' comes first
    aggregated_mean_percentage = aggregated_mean_percentage[['Total', 'Aware', 'Committed', 'Set Target']]
    
    # Create a new DataFrame for the output
    output_df = pd.concat([aggregated_mean, aggregated_mean_percentage], axis=1)
    
    # Rename the columns
    output_df.columns = pd.MultiIndex.from_product([['Number', 'Percentage'], ['Total', 'Aware', 'Committed', 'Set Target']])
    
    # Output the data
    print(output_df)

   

#%%Countries

    # Convert the dictionary columns into a multi-index DataFrame
    aggregated_data_expanded = pd.concat(
        [aggregated_data[col].apply(pd.Series) for col in ["TotalByCountry", "AwareByCountry", "CommittedByCountry", "SetTargetByCountry"]], 
        keys=["Total", "Aware", "Committed", "SetTarget"], 
        axis=1
    )
    
    # Extract the data for the last step
    last_step_total = aggregated_data_expanded["Total"].groupby(level=0).last()
    last_step_aware = aggregated_data_expanded["Aware"].groupby(level=0).last()
    last_step_committed = aggregated_data_expanded["Committed"].groupby(level=0).last()
    last_step_set_target = aggregated_data_expanded["SetTarget"].groupby(level=0).last()
    
        # Calculate means
    last_step_total_mean = last_step_total.mean()
    last_step_aware_mean = last_step_aware.mean()
    last_step_committed_mean = last_step_committed.mean()
    last_step_set_target_mean = last_step_set_target.mean()
    
    # Calculate stds
    last_step_total_std = last_step_total.std()
    last_step_aware_std = last_step_aware.std()
    last_step_committed_std = last_step_committed.std()
    last_step_set_target_std = last_step_set_target.std()
    
    # Calculate standard errors
    last_step_total_se = last_step_total_std / np.sqrt(len(results))
    last_step_aware_se = last_step_aware_std / np.sqrt(len(results))
    last_step_committed_se = last_step_committed_std / np.sqrt(len(results))
    last_step_set_target_se = last_step_set_target_std / np.sqrt(len(results))
    
    # Create a DataFrame from the mean series
    df_country_mean = pd.DataFrame({'Total': last_step_total_mean, 'Aware': last_step_aware_mean, 'Committed': last_step_committed_mean, 'Set Target': last_step_set_target_mean})
    
    # Create a DataFrame from the std series
    df_country_std = pd.DataFrame({'Total': last_step_total_std, 'Aware': last_step_aware_std, 'Committed': last_step_committed_std, 'Set Target': last_step_set_target_std})
    
    # Create a DataFrame from the se series
    df_country_se = pd.DataFrame({'Total': last_step_total_se, 'Aware': last_step_aware_se, 'Committed': last_step_committed_se, 'Set Target': last_step_set_target_se})
    
    # Choose top 10 countries based on mean total values
    top_10_countries = last_step_total_mean.nlargest(10).index
    
    # Filter the DataFrames to include only these countries
    df_country_mean_top_10 = df_country_mean.loc[top_10_countries]
    df_country_std_top_10 = df_country_std.loc[top_10_countries]
    df_country_se_top_10 = df_country_se.loc[top_10_countries]
    
    # Create a bar plot with error bars
    ax = df_country_mean_top_10.plot(kind="bar", yerr=df_country_std_top_10, figsize=(10, 7), capsize=4)
    #plt.title("Average Number of Companies in Each State for Top 10 Countries with SD")
    plt.ylabel("Number of Companies")
    plt.xlabel("Country")
    plt.ylim(bottom=0)  # Set lower limit to 0
    #plt.savefig("images_experiments/base_scenario_2233_top10countries.png")
    plt.show()
    

    #%%Countries step 60
    
    # Convert the dictionary columns into a multi-index DataFrame
    aggregated_data_expanded = pd.concat(
        [aggregated_data[col].apply(pd.Series) for col in ["TotalByCountry", "AwareByCountry", "CommittedByCountry", "SetTargetByCountry"]], 
        keys=["Total", "Aware", "Committed", "SetTarget"], 
        axis=1
    )
    
    # Extract the data for the 60th step
    step_60_total = aggregated_data_expanded["Total"].groupby(level=0).apply(lambda x: x.iloc[59])
    step_60_aware = aggregated_data_expanded["Aware"].groupby(level=0).apply(lambda x: x.iloc[59])
    step_60_committed = aggregated_data_expanded["Committed"].groupby(level=0).apply(lambda x: x.iloc[59])
    step_60_set_target = aggregated_data_expanded["SetTarget"].groupby(level=0).apply(lambda x: x.iloc[59])

    # Calculate means
    step_60_total_mean = step_60_total.mean()
    step_60_aware_mean = step_60_aware.mean()
    step_60_committed_mean = step_60_committed.mean()
    step_60_set_target_mean = step_60_set_target.mean()
    
    # Calculate stds
    step_60_total_std = step_60_total.std()
    step_60_aware_std = step_60_aware.std()
    step_60_committed_std = step_60_committed.std()
    step_60_set_target_std = step_60_set_target.std()
    
    # Calculate standard errors
    step_60_total_se = step_60_total_std / np.sqrt(len(results))
    step_60_aware_se = step_60_aware_std / np.sqrt(len(results))
    step_60_committed_se = step_60_committed_std / np.sqrt(len(results))
    step_60_set_target_se = step_60_set_target_std / np.sqrt(len(results))
    
    # Create a DataFrame from the mean series
    df_country_mean = pd.DataFrame({'Total': step_60_total_mean, 'Committed': step_60_committed_mean, 'Set Target': step_60_set_target_mean})
    
    # Create a DataFrame from the std series
    df_country_std = pd.DataFrame({'Total': step_60_total_std, 'Committed': step_60_committed_std, 'Set Target': step_60_set_target_std})
    
    # Create a DataFrame from the se series
    df_country_se = pd.DataFrame({'Total': step_60_total_se, 'Committed': step_60_committed_se, 'Set Target': step_60_set_target_se})
    
    
    print(df_country_mean)
    df_country_mean.to_csv('data_collected/country_mean.csv', index='Country')
    # Choose top 10 countries based on mean total values
    top_10_countries = step_60_total_mean.nlargest(10).index
    
    # Filter the DataFrames to include only these countries
    df_country_mean_top_10 = df_country_mean.loc[top_10_countries]
    df_country_std_top_10 = df_country_std.loc[top_10_countries]
    df_country_se_top_10 = df_country_se.loc[top_10_countries]
    
    # Create a bar plot with error bars
    ax = df_country_mean_top_10.plot(kind="bar", yerr=df_country_std_top_10, figsize=(14, 10), capsize=4, color=['skyblue', 'mediumseagreen', 'lightcoral'])
    #plt.title("Average Number of Companies in Each State for Top 10 Countries with SD")
    plt.ylabel("Number of Companies", fontsize=14)
    plt.xlabel("Country", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)  # Set lower limit to 0
    #plt.savefig("report_images/base_scenario_countries.png", dpi = 500, bbox_inches='tight')
    plt.show()



#%%




#%%

    
    # Collect culture dimensions data
    aware_data = [result[3][["AverageCommunicatingAware", "AverageEvaluatingAware", "AverageLeadingAware", "AverageDecidingAware", "AverageTrustingAware", "AverageDisagreeingAware", "AverageSchedulingAware"]].iloc[-1] for result in results]
    committed_data = [result[3][["AverageCommunicatingCommitted", "AverageEvaluatingCommitted", "AverageLeadingCommitted", "AverageDecidingCommitted", "AverageTrustingCommitted", "AverageDisagreeingCommitted", "AverageSchedulingCommitted"]].iloc[-1] for result in results]
    target_data = [result[3][["AverageCommunicatingSetTarget", "AverageEvaluatingSetTarget", "AverageLeadingSetTarget", "AverageDecidingSetTarget", "AverageTrustingSetTarget", "AverageDisagreeingSetTarget", "AverageSchedulingSetTarget"]].iloc[-1] for result in results]

    # Convert to DataFrame
    aware_df = pd.DataFrame(aware_data)
    committed_df = pd.DataFrame(committed_data)
    target_df = pd.DataFrame(target_data)
    
    # Calculate means
    aware_means = aware_df.mean()
    committed_means = committed_df.mean()
    target_means = target_df.mean()
    
    # Calculate stds
    aware_stds = aware_df.std()
    committed_stds = committed_df.std()
    target_stds = target_df.std()
    
    my_labels = ["Communicating", "Evaluating", "Leading", "Deciding", "Trusting", "Disagreeing", "Scheduling"]
    
    def grouped_bar_chart(labels, means1, means2, means3, stds1, stds2, stds3):
        x = np.arange(len(labels))  # the label locations
        width = 0.3  # the width of the bars
    
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width, means1, width, label='Aware Companies', yerr=stds1, capsize=4)
        rects2 = ax.bar(x, means2, width, label='Committed Companies', yerr=stds2, capsize=4)
        rects3 = ax.bar(x + width, means3, width, label='Companies With Target', yerr=stds3, capsize=4)
    
        ax.set_xlabel('Cultural Dimensions')
        ax.set_ylabel('Average Value')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', borderaxespad=0., ncol=3)  # Moves the legend outside the plot
    
        fig.tight_layout()
        #plt.savefig("images_experiments/base_scenario_2233_culture.png")
        plt.show()
    
    grouped_bar_chart(
        my_labels, 
        aware_means.values, 
        committed_means.values, 
        target_means.values,
        aware_stds.values, 
        committed_stds.values, 
        target_stds.values, 
    )
    
    
#%% Culture dimensions step 60
    # Filter to only include 'BaseScenario' results
    results_copy = [result for result in results if result[0] == 'BaseScenario']
    
    # Initialize empty lists to collect data
    aware_data = []
    committed_data = []
    target_data = []
    
    # Extract the data_model for each result and get the data for the 60th step
    for result in results_copy:
        step_60_data = result[3].iloc[59]  # Assuming index starts at 0, 59 corresponds to step 60
    
        aware_data.append(step_60_data[["AverageCommunicatingAware", "AverageEvaluatingAware", "AverageLeadingAware", "AverageDecidingAware", "AverageTrustingAware", "AverageDisagreeingAware", "AverageSchedulingAware"]])
        committed_data.append(step_60_data[["AverageCommunicatingCommitted", "AverageEvaluatingCommitted", "AverageLeadingCommitted", "AverageDecidingCommitted", "AverageTrustingCommitted", "AverageDisagreeingCommitted", "AverageSchedulingCommitted"]])
        target_data.append(step_60_data[["AverageCommunicatingSetTarget", "AverageEvaluatingSetTarget", "AverageLeadingSetTarget", "AverageDecidingSetTarget", "AverageTrustingSetTarget", "AverageDisagreeingSetTarget", "AverageSchedulingSetTarget"]])
    
    # Convert to DataFrame
    aware_df = pd.DataFrame(aware_data)
    committed_df = pd.DataFrame(committed_data)
    target_df = pd.DataFrame(target_data)
    # Calculate means
    aware_means = aware_df.mean()
    committed_means = committed_df.mean()
    target_means = target_df.mean()
    
    # Calculate stds
    aware_stds = aware_df.std()
    committed_stds = committed_df.std()
    target_stds = target_df.std()
    
    my_labels = ["Communicating", "Evaluating", "Leading", "Deciding", "Trusting", "Disagreeing", "Scheduling"]
    
    def grouped_bar_chart(labels, means1, means2, means3, stds1, stds2, stds3):
        x = np.arange(len(labels))  # the label locations
        width = 0.3  # the width of the bars
    
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width, means1*100, width, label='Total Companies', yerr=stds1*100, capsize=4, color='skyblue')
        rects2 = ax.bar(x, means2*100, width, label='Committed Companies', yerr=stds2*100, capsize=4, color='mediumseagreen')
        rects3 = ax.bar(x + width, means3*100, width, label='Companies With Target', yerr=stds3*100, capsize=4, color='lightcoral')
    
        ax.set_xlabel('Cultural Dimensions')
        ax.set_ylabel('Average Value')
    
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', borderaxespad=0., ncol=3)  # Moves the legend outside the plot
    
        fig.tight_layout()
        plt.show()
    
    grouped_bar_chart(
        my_labels, 
        aware_means.values, 
        committed_means.values, 
        target_means.values,
        aware_stds.values, 
        committed_stds.values, 
        target_stds.values, 
    )
    
    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Cultural_Dimensions': my_labels,
        'Total_Mean': aware_means.values,  # Replace this with the correct values
        'Total_Std': aware_stds.values,  # Replace this with the correct values
        'Committed_Mean': committed_means.values,
        'Committed_Std': committed_stds.values,
        'Set_Target_Mean': target_means.values,
        'Set_Target_Std': target_stds.values
    })
    
    # Save the DataFrame to a CSV file
    #results_df.to_csv('data_collected/culture_dimensions_results.csv', index=False)
    
    # Print the DataFrame to the console
    print(results_df)
    
    


#%%
    from collections import defaultdict
    
    # Initialize a dictionary to hold the sums
    scenario_data = defaultdict(list)
    
    for scenario, params_set, run_number, data_model in results:
        scenario_data[scenario].append(data_model["AverageConnections"])
    
    for scenario, data_list in scenario_data.items():
        scenario_data[scenario] = pd.concat(data_list, axis=1)
    
    import matplotlib.pyplot as plt

    for scenario, data in scenario_data.items():
        mean_values = data.mean(axis=1)
        std_values = data.std(axis=1)
    
        plt.figure(figsize=(10, 6))
        mean_values.plot(label="Mean")
        plt.fill_between(data.index, mean_values - std_values, mean_values + std_values, color='b', alpha=0.2, label="Standard Deviation")
        
        plt.title(f'Average Connections for Scenario: {scenario}')
        plt.xlabel('Steps')
        plt.ylabel('Average Connections')
        plt.legend()
        plt.grid(True)
        plt.show()




