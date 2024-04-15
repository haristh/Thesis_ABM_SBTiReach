#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:57:12 2023

@author: charistheodorou


This file contains all the graphs and tables used in the Results and Experimentation
Chapter.


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





#%% BASE SCENARIO
number_of_steps = 60
num_companies = 2233

#
with open('data_collected/results_BaseScenario_2233.pickle', 'rb') as f:
    results = pickle.load(f)


#%% Progress over time 

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
plt.xlabel('Time')
plt.ylabel('Percentage of companies (%)')
plt.grid(True)
plt.xlim([datetime.date(2015, 6, 1), datetime.date(2020, 5, 1)])
#plt.savefig("images_experiments/base_scenario_2233_progress.png")
plt.show()

#%% average time

# Extract the "AverageCommitToTargetTime" from the final step of each run
average_commit_to_target_time_data = [result[3]["AverageCommitToTargetTime"].iloc[-1] for result in results]

# Calculate the mean value across all runs
average_commit_to_target_time_mean = np.mean(average_commit_to_target_time_data)

# Calculate the standard deviation
average_commit_to_target_time_std = np.std(average_commit_to_target_time_data)

# Calculate the standard error
average_commit_to_target_time_se = average_commit_to_target_time_std / np.sqrt(len(average_commit_to_target_time_data))

print(f"Average of 'AverageCommitToTargetTime' in the last step across all runs: {average_commit_to_target_time_mean} ± {average_commit_to_target_time_se}")


#%% Average final number of aware/committed/with target companies after 60 steps

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

#%%     Sectors
# Extract the model data from the results
model_data = [result[3] for result in results]  # result[3] is the data_model

# Aggregate the collected model data
aggregated_data = pd.concat(model_data, keys=range(len(model_data)))
# Group the data by the first level of the index (which is the run number) and get the last entry of each group
last_step_data = aggregated_data.groupby(level=0).last()
    

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
#plt.savefig("images_experiments/base_scenario_2233_sectors.png")
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
    plt.savefig("images_experiments/base_scenario_2233_culture.png")
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



#%% FIRST SCENARIOS

scenarios = {
    'ShareholderPressure':  [(5,1,1,1), (10,1,1,1)],
    'ManagerPressure':      [(1,5,1,1), (1,10,1,1)],
    'EmployeePressure':     [(1,1,5,1), (1,1,10,1)],
    'MarketPressure':       [(1,1,1,5), (1,1,1,10)],
    'AllPressure':          [(5,5,5,5), (10,10,10,10)],
    'NoIncrease':           [(1,1,1,1)]
}

with open('data_collected/all_model_results_2233.pickle', 'rb') as f:
    all_model_results = pickle.load(f)

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
    axs[0].plot(averaged_results[scenario][params_index].index[63:], 
                averaged_results[scenario][params_index]['Committed'][63:], 
                color=color, label=scenario)
    axs[0].fill_between(averaged_results[scenario][params_index].index[63:],
                        (averaged_results[scenario][params_index]['Committed'] - std_dev_results[scenario][params_index]['Committed'])[63:],
                        (averaged_results[scenario][params_index]['Committed'] + std_dev_results[scenario][params_index]['Committed'])[63:],
                        color='gray', alpha=0.5)
    
    # Set Targets subplot
    axs[1].plot(averaged_results[scenario][params_index].index[:64], 
                averaged_results[scenario][params_index]['HasTarget'][:64], 
                color='red')
    axs[1].fill_between(averaged_results[scenario][params_index].index[:64],
                        (averaged_results[scenario][params_index]['HasTarget'] - std_dev_results[scenario][params_index]['HasTarget'])[:64],
                        (averaged_results[scenario][params_index]['HasTarget'] + std_dev_results[scenario][params_index]['HasTarget'])[:64],
                        color='gray', alpha=0.5)
    axs[1].plot(averaged_results[scenario][params_index].index[63:], 
                averaged_results[scenario][params_index]['HasTarget'][63:], 
                color=color, label=scenario)
    axs[1].fill_between(averaged_results[scenario][params_index].index[63:],
                        (averaged_results[scenario][params_index]['HasTarget'] - std_dev_results[scenario][params_index]['HasTarget'])[63:],
                        (averaged_results[scenario][params_index]['HasTarget'] + std_dev_results[scenario][params_index]['HasTarget'])[63:],
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




differences = {
    scenario: {
        "CommitmentIncrease": averaged_results[scenario][param_sets[scenario].index(desired_params[scenario])]['Committed'].iloc[-1] - averaged_results[scenario][param_sets[scenario].index(desired_params[scenario])]['Committed'].iloc[63],
        "SetTargetIncrease": averaged_results[scenario][param_sets[scenario].index(desired_params[scenario])]['HasTarget'].iloc[-1] - averaged_results[scenario][param_sets[scenario].index(desired_params[scenario])]['HasTarget'].iloc[63]
    }
    for scenario in color_map.keys()
}

df_differences = pd.DataFrame(differences).T/num_companies
print(df_differences)
   
    


#%% SECOND SCENARIOS - needs to be turned into percentages

number_of_steps = 108
num_companies = 2233

# load
with open('data_collected/results_secondsetsenarios_2233.pickle', 'rb') as f:
    results = pickle.load(f)
     
     

#%% Progress over time for commitments and set targets
from matplotlib.dates import date2num, num2date


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
    'MarketPressureCampaign': 'blue',
#    'ManagerPressure': 'pink',
    'LongerDeadline':'purple',
#    'ShorterDeadline':'orange',
}



# Convert the step numbers to dates
start_date = pd.to_datetime('2015-05-01')
date_index = pd.date_range(start=start_date, periods=number_of_steps, freq='M')

# Create the first plot for 'Commitments Over Time'
#plt.figure(figsize=(10, 7))
fig1, ax1 = plt.subplots(figsize=(10, 7))

# Plotting Commitments Over Time
for scenario in color_map.keys():
    color = color_map[scenario]
    label = "NoCampaigns" if scenario == "BaseScenario" else scenario  # Custom label
    plt.plot(date_index[:64],
             (averaged_results[scenario]['Committed'][:64] / num_companies) * 100,
             color='green')
    plt.fill_between(date_index[:64],
                     ((averaged_results[scenario]['Committed'] - std_dev_results[scenario]['Committed']) / num_companies)[:64] * 100,
                     ((averaged_results[scenario]['Committed'] + std_dev_results[scenario]['Committed']) / num_companies)[:64] * 100,
                     color='gray', alpha=0.5)
    plt.plot(date_index[63:],
             (averaged_results[scenario]['Committed'][63:] / num_companies) * 100,
             color=color, label=label)
    plt.fill_between(date_index[63:],
                     ((averaged_results[scenario]['Committed'] - std_dev_results[scenario]['Committed']) / num_companies)[63:] * 100,
                     ((averaged_results[scenario]['Committed'] + std_dev_results[scenario]['Committed']) / num_companies)[63:] * 100,
                     color='gray', alpha=0.5)

#plt.title('Commitments Over Time')
plt.xlabel('Time step', fontsize= 14)
plt.ylabel('Percentage of Companies (%)', fontsize = 14)
ax1.axvline(x=date2num(start_date + pd.DateOffset(months=64)), color='grey', linestyle='--') 
plt.legend(fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('report/scenarios_set2_committed.png', dpi= 150)

plt.show()

# Create the second plot for 'Set Targets Over Time'
fig1, ax1 = plt.subplots(figsize=(10, 7))

# Plotting Set Targets Over Time
for scenario in color_map.keys():
    color = color_map[scenario]
    label="NoCampaigns" if scenario == "BaseScenario" else scenario  # Custom label
    plt.plot(date_index[:64],
             (averaged_results[scenario]['HasTarget'][:64] / num_companies) * 100,
             color='red')
    plt.fill_between(date_index[:64],
                     ((averaged_results[scenario]['HasTarget'] - std_dev_results[scenario]['HasTarget']) / num_companies)[:64] * 100,
                     ((averaged_results[scenario]['HasTarget'] + std_dev_results[scenario]['HasTarget']) / num_companies)[:64] * 100,
                     color='gray', alpha=0.5)
    plt.plot(date_index[63:],
             (averaged_results[scenario]['HasTarget'][63:] / num_companies) * 100,
             color=color, label=label)
    plt.fill_between(date_index[63:],
                     ((averaged_results[scenario]['HasTarget'] - std_dev_results[scenario]['HasTarget']) / num_companies)[63:] * 100,
                     ((averaged_results[scenario]['HasTarget'] + std_dev_results[scenario]['HasTarget']) / num_companies)[63:] * 100,
                     color='gray', alpha=0.5)

#plt.title('Set Targets Over Time')
plt.xlabel('Time step', fontsize =14)
plt.ylabel('Percentage of Companies (%)', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ax1.axvline(x=date2num(start_date + pd.DateOffset(months=64)), color='grey', linestyle='--') 
plt.legend(fontsize = 12)
plt.savefig('report/scenarios_set2_targets.png', dpi= 150)
plt.show()


#%%
fig, axs = plt.subplots(2, figsize=(10, 15))  # Create two subplots

# Plot lines for each scenario with standard deviation as gray shade
for scenario in color_map.keys():
    color = color_map[scenario]  # Fetch the color for this scenario

    # Committed subplot
    axs[0].plot(date_index[:64],
                averaged_results[scenario]['Committed'][:64],
                color='blue')
    axs[0].fill_between(date_index[:64],
                        (averaged_results[scenario]['Committed'] - std_dev_results[scenario]['Committed'])[:64],
                        (averaged_results[scenario]['Committed'] + std_dev_results[scenario]['Committed'])[:64],
                        color='gray', alpha=0.5)
    axs[0].plot(date_index[63:],
                averaged_results[scenario]['Committed'][63:],
                color=color, label=scenario)
    axs[0].fill_between(date_index[63:],
                        (averaged_results[scenario]['Committed'] - std_dev_results[scenario]['Committed'])[63:],
                        (averaged_results[scenario]['Committed'] + std_dev_results[scenario]['Committed'])[63:],
                        color='gray', alpha=0.5)

    # Set Targets subplot
    axs[1].plot(date_index[:64],
                averaged_results[scenario]['HasTarget'][:64],
                color='red')
    axs[1].fill_between(date_index[:64],
                        (averaged_results[scenario]['HasTarget'] - std_dev_results[scenario]['HasTarget'])[:64],
                        (averaged_results[scenario]['HasTarget'] + std_dev_results[scenario]['HasTarget'])[:64],
                        color='gray', alpha=0.5)
    axs[1].plot(date_index[63:],
                averaged_results[scenario]['HasTarget'][63:],
                color=color, label=scenario)
    axs[1].fill_between(date_index[63:],
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

differences = {
    scenario: {
        "CommitmentIncrease": averaged_results[scenario]['Committed'].iloc[-1] - averaged_results[scenario]['Committed'].iloc[63],
        "SetTargetIncrease": averaged_results[scenario]['HasTarget'].iloc[-1] - averaged_results[scenario]['HasTarget'].iloc[63]
    }
    for scenario in color_map.keys()
}

df_differences = pd.DataFrame(differences).T/num_companies
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
plt.xlabel('Time')
plt.ylabel('Percentage of companies (%)')
plt.grid(True)
plt.xlim([datetime.date(2015, 6, 1), datetime.date(2020, 5, 1)])
#plt.savefig("images_experiments/base_scenario_2233_progress.png")
plt.show()


#%%



from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# Function to convert percentage string to float
def percentage_to_float(percentage_str):
    try:
        return float(percentage_str.strip('%'))
    except:
        return None

# Read Excel file
df = pd.read_excel('files/StructuredData.xlsx', sheet_name='Countries_results')
print(df.columns)
#%%



# Drop rows with any NaN values for accurate calculations
df = df.dropna(subset=['Committed %', 'Set Target %', 'Model Committed %', 'Model Set Target %'])



# Function to calculate Pearson r and MAE
def calc_metrics(df, col1, col2):
    corr, _ = pearsonr(df[col1], df[col2])
    mae = mean_absolute_error(df[col1], df[col2])
    return corr, mae

# For All Countries
corr_committed_all, mae_committed_all = calc_metrics(df, 'Committed %', 'Model Committed %')
corr_set_target_all, mae_set_target_all = calc_metrics(df, 'Set Target %', 'Model Set Target %')

print(f'All countries - Pearson r for committed: {corr_committed_all}, MAE for committed: {mae_committed_all}')
print(f'All countries - Pearson r for set target: {corr_set_target_all}, MAE for set target: {mae_set_target_all}')

# For the first 30 most represented countries (you might need to sort df by 'Total' or other criteria if needed)
df_top30 = df.nlargest(30, 'Total')

corr_committed_30, mae_committed_30 = calc_metrics(df_top30, 'Committed %', 'Model Committed %')
corr_set_target_30, mae_set_target_30 = calc_metrics(df_top30, 'Set Target %', 'Model Set Target %')

print(f'Top 30 countries - Pearson r for committed: {corr_committed_30}, MAE for committed: {mae_committed_30}')
print(f'Top 30 countries - Pearson r for set target: {corr_set_target_30}, MAE for set target: {mae_set_target_30}')

#%%
from scipy.stats import pearsonr
from numpy import average
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd


# Read Excel file
df = pd.read_excel('files/StructuredData.xlsx', sheet_name='Countries_results')

# Drop rows with any NaN values for accurate calculations
df = df.dropna(subset=['Committed %', 'Set Target %', 'Model Committed %', 'Model Set Target %'])



# Function to calculate weighted Pearson r and weighted MAE
def calc_weighted_metrics(df, col1, col2, weight_col):
    weights = df[weight_col]
    
    # Calculate weighted Pearson r
    weighted_cov = np.sum(weights * (df[col1] - average(df[col1], weights=weights)) * (df[col2] - average(df[col2], weights=weights))) / np.sum(weights)
    weighted_var1 = np.sum(weights * (df[col1] - average(df[col1], weights=weights)) ** 2) / np.sum(weights)
    weighted_var2 = np.sum(weights * (df[col2] - average(df[col2], weights=weights)) ** 2) / np.sum(weights)
    
    weighted_corr = weighted_cov / np.sqrt(weighted_var1 * weighted_var2)
    
    # Calculate weighted MAE
    weighted_mae = np.sum(weights * np.abs(df[col1] - df[col2])) / np.sum(weights)
    
    return weighted_corr, weighted_mae

# For All Countries
weighted_corr_committed_all, weighted_mae_committed_all = calc_weighted_metrics(df, 'Committed %', 'Model Committed %', 'Total')
weighted_corr_set_target_all, weighted_mae_set_target_all = calc_weighted_metrics(df, 'Set Target %', 'Model Set Target %', 'Total')

print(f'All countries - Weighted Pearson r for committed: {weighted_corr_committed_all}, Weighted MAE for committed: {weighted_mae_committed_all}')
print(f'All countries - Weighted Pearson r for set target: {weighted_corr_set_target_all}, Weighted MAE for set target: {weighted_mae_set_target_all}')

# Sort the DataFrame based on the 'Total' column and take the top 30 countries
df_top30 = df.nlargest(30, 'Total')

# For the top 30 most represented countries
weighted_corr_committed_30, weighted_mae_committed_30 = calc_weighted_metrics(df_top30, 'Committed %', 'Model Committed %', 'Total')
weighted_corr_set_target_30, weighted_mae_set_target_30 = calc_weighted_metrics(df_top30, 'Set Target %', 'Model Set Target %', 'Total')

print(f'Top 30 countries - Weighted Pearson r for committed: {weighted_corr_committed_30}, Weighted MAE for committed: {weighted_mae_committed_30}')
print(f'Top 30 countries - Weighted Pearson r for set target: {weighted_corr_set_target_30}, Weighted MAE for set target: {weighted_mae_set_target_30}')



#%%

# Function to calculate weighted Pearson r and MAE
def calc_weighted_metrics(df, col1, col2, weight_col):
    weighted_mean1 = np.average(df[col1], weights=df[weight_col])
    weighted_mean2 = np.average(df[col2], weights=df[weight_col])

    # Calculate weighted Pearson correlation
    numerator = np.sum(df[weight_col] * (df[col1] - weighted_mean1) * (df[col2] - weighted_mean2))
    denominator = np.sqrt(np.sum(df[weight_col] * (df[col1] - weighted_mean1)**2) * np.sum(df[weight_col] * (df[col2] - weighted_mean2)**2))
    
    weighted_corr = numerator / denominator

    # Calculate weighted MAE
    weighted_mae = np.sum(df[weight_col] * np.abs(df[col1] - df[col2])) / np.sum(df[weight_col])

    return weighted_corr, weighted_mae

# Load data from Excel
df = pd.read_excel("files/StructuredData.xlsx", sheet_name="Sectors_results")


# Drop rows with any NaN values for accurate calculations
df = df.dropna(subset=['Committed %', 'Set Target %', 'Model Committed %', 'Model Set Target %'])

# Calculate weighted metrics
weighted_corr_committed, weighted_mae_committed = calc_weighted_metrics(df, 'Committed %', 'Model Committed %', 'Number')
weighted_corr_set_target, weighted_mae_set_target = calc_weighted_metrics(df, 'Set Target %', 'Model Set Target %', 'Number')

print(f"Sectors - Weighted Pearson r for committed: {weighted_corr_committed}, Weighted MAE for committed: {weighted_mae_committed}")
print(f"Sectors - Weighted Pearson r for set target: {weighted_corr_set_target}, Weighted MAE for set target: {weighted_mae_set_target}")



# Scatter plot for Committed %
plt.figure(figsize=(10, 6))
plt.scatter(df['Committed %'], df['Model Committed %'], c='blue')
plt.title('Scatter Plot for Committed %')
plt.xlabel('Actual Committed %')
plt.ylabel('Model Committed %')
plt.grid(True)
plt.show()

# Scatter plot for Set Target %
plt.figure(figsize=(10, 6))
plt.scatter(df['Set Target %'], df['Model Set Target %'], c='green')
plt.title('Scatter Plot for Set Target %')
plt.xlabel('Actual Set Target %')
plt.ylabel('Model Set Target %')
plt.grid(True)
plt.show()

# Remove outliers based on a condition
df_filtered = df[df['Committed %'] <= 0.6]

# Calculate Pearson r for the filtered data
pearson_r_committed, _ = pearsonr(df_filtered['Committed %'], df_filtered['Model Committed %'])
print(f"Filtered Pearson r for committed: {pearson_r_committed}")
# Calculate weighted metrics
weighted_corr_committed, weighted_mae_committed = calc_weighted_metrics(df_filtered, 'Committed %', 'Model Committed %', 'Number')
weighted_corr_set_target, weighted_mae_set_target = calc_weighted_metrics(df_filtered, 'Set Target %', 'Model Set Target %', 'Number')
print(f"Sectors - Weighted Pearson r for committed: {weighted_corr_committed}, Weighted MAE for committed: {weighted_mae_committed}")
print(f"Sectors - Weighted Pearson r for set target: {weighted_corr_set_target}, Weighted MAE for set target: {weighted_mae_set_target}")



# Scatter plot for Committed %
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Committed %'], df_filtered['Model Committed %'], c='blue')
plt.title('Scatter Plot for Committed %')
plt.xlabel('Actual Committed %')
plt.ylabel('Model Committed %')
plt.grid(True)
plt.show()

# Scatter plot for Set Target %
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Set Target %'], df_filtered['Model Set Target %'], c='green')
plt.title('Scatter Plot for Set Target %')
plt.xlabel('Actual Set Target %')
plt.ylabel('Model Set Target %')
plt.grid(True)
plt.show()