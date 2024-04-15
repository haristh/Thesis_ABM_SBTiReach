#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:20:33 2023

@author: charistheodorou

This file runs the SBTi model once and presents its results. Used mainly for fast and easy 
verification that model works as expected.


Running the data collection, saving figures and saving any data is commented out to avoid overwriting


"""

import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SBTiModel_main import SBTiModel
import os




num_companies = 100
number_of_steps = 60

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
motivation_coefficient = 4.08 # After running the model, maximum motivation value ends up being around 2.2
pressure_coefficient = 5.02
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




#%% Extraction of data
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



#%%

# Dictionary with the initial agent parameters
initial_agent_df = pd.DataFrame(model.initial_agent_data)

# start the steps of the model
for _ in range(number_of_steps):
    model.step(shareholder_pressure_lever,
               manager_pressure_lever,
               employee_pressure_lever,
               market_pressure_lever)
   
#
# Data Collectors/Reporters
data_model = model.datacollector.get_model_vars_dataframe()



# convert the step numbers to dates starting from September 2015- Εach step is a month
start_date = pd.to_datetime('2015-09-01')
data_model.index = pd.date_range(start=start_date, periods=number_of_steps, freq='M')


# Calling the main outcomes % commitment, % set targets and average time to set target
avg_time = model.average_time_to_set_target()
commitment_percentage =  model.get_committed_total_percent()
target_set_percentage = model.get_target_set_total_percent()

if avg_time is not None:
    print(f"Average time to set target after {number_of_steps} steps: {avg_time:.2f} months")
    print(f"Percentage of companies that committed after {number_of_steps} steps: {commitment_percentage:.2f}")
    print(f"Percentage of companies that set target after {number_of_steps} steps: {target_set_percentage:.2f}")


#%% Aware/Committed/HasTargets numbers through time 2015-2030

# Create the folder if it doesn't exist
folder_name = "model_images"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)




# create the plot
plt.figure(figsize=(10, 6))
plt.plot(data_model['Aware'], label='Aware', color='blue')
plt.plot(data_model['Committed'], label='Committed', color='green')
plt.plot(data_model['HasTarget'], label='HasTarget', color='red')
#plt.plot(data['NotAware'], label='NotAware', color='purple')
plt.legend(loc='best')
plt.title('Evolution of company states over time')
plt.xlabel('Time')
plt.ylabel('Number of companies')
plt.grid(True)
plt.show()


#%% Final results per sector

# Suppose your DataCollector collected model-level variable "CommittedBySector" at the end of each run
# Get the model-level data
model_data = model.datacollector.get_model_vars_dataframe()

# Extract the data for the last step
last_step_total = pd.Series(model_data["TotalBySector"].iloc[-1])
last_step_aware = pd.Series(model_data["AwareBySector"].iloc[-1])
last_step_committed = pd.Series(model_data["CommittedBySector"].iloc[-1])
last_step_set_target = pd.Series(model_data["SetTargetBySector"].iloc[-1])

# Create a DataFrame from the series
df_sector = pd.DataFrame({'Total': last_step_total, 'Aware': last_step_aware, 'Committed': last_step_committed, 'Set Target': last_step_set_target})

# Create a bar plot
df_sector.plot(kind="bar", figsize=(10, 7))
plt.title("Number of Companies in Each State per Sector")
plt.ylabel("Number of Companies")
plt.xlabel("Sector")

plt.show()


#%%Final results per country


model_data = model.datacollector.get_model_vars_dataframe()

# Get the data of the last step for each measure
last_step_aware = pd.DataFrame(model_data["AwareByCountry"].iloc[-1].items(), columns=["Country", "Aware"])
last_step_committed = pd.DataFrame(model_data["CommittedByCountry"].iloc[-1].items(), columns=["Country", "Committed"])
last_step_set_target = pd.DataFrame(model_data["SetTargetByCountry"].iloc[-1].items(), columns=["Country", "Set Target"])
last_step_total = pd.DataFrame(model_data["TotalByCountry"].iloc[-1].items(), columns=["Country", "Total"])

# Merge the data into a single DataFrame
df_country = pd.merge(last_step_total, last_step_aware, on="Country")
df_country = pd.merge(df_country, last_step_committed, on="Country")
df_country = pd.merge(df_country, last_step_set_target, on="Country")

# Sort the total_companies DataFrame in descending order and select the first 5 rows
df_country_top10 = df_country.sort_values(by="Total", ascending=False).head(10)

# Plot the data
df_country_top10.plot(x="Country", y=["Total", "Aware", "Committed", "Set Target"], kind="bar", figsize=(10, 7))
plt.title("Number of Companies in Each State per Country")
plt.ylabel("Number of Companies")
plt.xlabel("Country")
plt.show()


#%%
# Average Culture Dimensions - Aware Companies:
average_culture_dimensions_aware = model_data.iloc[-1][[
    "AverageCommunicatingAware",
    "AverageEvaluatingAware",
    "AverageLeadingAware",
    "AverageDecidingAware",
    "AverageTrustingAware",
    "AverageDisagreeingAware",
    "AverageSchedulingAware",
]]

print("Average Culture Dimensions - Aware Companies:")
print(average_culture_dimensions_aware)

# Average Culture Dimensions - Committed Companies:
average_culture_dimensions_committed = model_data.iloc[-1][[
    "AverageCommunicatingCommitted",
    "AverageEvaluatingCommitted",
    "AverageLeadingCommitted",
    "AverageDecidingCommitted",
    "AverageTrustingCommitted",
    "AverageDisagreeingCommitted",
    "AverageSchedulingCommitted",
]]

print("\nAverage Culture Dimensions - Committed Companies:")
print(average_culture_dimensions_committed)

# Average Culture Dimensions - Companies With Target:
average_culture_dimensions_target = model_data.iloc[-1][[
    "AverageCommunicatingSetTarget",
    "AverageEvaluatingSetTarget",
    "AverageLeadingSetTarget",
    "AverageDecidingSetTarget",
    "AverageTrustingSetTarget",
    "AverageDisagreeingSetTarget",
    "AverageSchedulingSetTarget",
]]

print("\nAverage Culture Dimensions - Companies With Target:")
print(average_culture_dimensions_target)

my_labels = ["Communicating", "Evaluating", "Leading", "Deciding", "Trusting", "Disagreeing", "Scheduling"]

# Define a function to create a grouped bar chart
def grouped_bar_chart(labels, values1, values2, values3, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width, values1, width, label='Aware Companies')
    rects2 = ax.bar(x, values2, width, label='Committed Companies')
    rects3 = ax.bar(x + width, values3, width, label='Companies With Target')

    ax.set_xlabel('Cultural Dimensions')
    ax.set_ylabel('Average Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', borderaxespad=0., ncol=3)  # Moves the legend outside the plot

    fig.tight_layout()
    
    plt.show()

# Create a grouped bar chart
grouped_bar_chart(
    my_labels, 
    average_culture_dimensions_aware.values, 
    average_culture_dimensions_committed.values,
    average_culture_dimensions_target.values, 
    'Comparison of Average Cultural Dimensions'
)


#%% List of failed companies to set target after committing
failed_agents = model.get_agents_failed_to_set_target()

# convert to a dataframe if desired
failed_agents_df = pd.DataFrame(failed_agents.values(), index=failed_agents.keys())

print("List of failed companies to set target after committing: ",failed_agents_df)






