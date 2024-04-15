#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:20:33 2023

@author: charistheodorou

This file runs the SBTi model once and presents its results. Used mainly for fast and easy 
verification that model works as expected. The extra part in this one is the failed companies data.


Running the data collection, saving figures and saving any data is commented out to avoid overwriting



"""

import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SBTiModel_scenarios_set2 import SBTiModel
import os




num_companies = 200
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

connectivity_lever = 13
market_pressure_on = False
meetings_per_year_scenario = 10 
max_comm_duration_scenario = 24
#%% Sanity Chacks

# y = np.random.normal(mu_leading.loc['USA'], 2.5)
# print(y)
# print(sector_probs)
# print(emissions)

#country= 'USA'
#x = float(mu_scheduling.loc[country])
#print(x)


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
    

    connectivity_lever = connectivity_lever,
    market_pressure_on = market_pressure_on,
    
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
               market_pressure_lever, connectivity_lever,
               meetings_per_year_scenario, max_comm_duration_scenario)
   
# Model visualisation-  mostly unnecessary
#model.visualize_network()

# Data Collectors/Reporters
data_model = model.datacollector.get_model_vars_dataframe()
data_agent = model.datacollector.get_agent_vars_dataframe()


# Representation of data
ConnectionMatrix = model.M
#InitialConnections = pd.DataFrame(model.initial_connections_data)
#M_formatted = np.array2string(model.M, formatter={'float_kind':'{:.1f}'.format}) 



# convert the step numbers to dates starting from September 2015- Εach step is a month
start_date = pd.to_datetime('2015-09-01')
data_model.index = pd.date_range(start=start_date, periods=number_of_steps, freq='M')


# Two ways of giving average time. Calling it from datacollectors is one step off because it doesn't take into consideration the last step

avg_time = model.average_time_to_set_target()
commitment_percentage =  model.get_committed_total_percent()
target_set_percentage = model.get_target_set_total_percent()

if avg_time is not None:
    print(f"Average time to set target after {number_of_steps} steps: {avg_time:.2f} months")
    print(f"Percentage of companies that committed after {number_of_steps} steps: {commitment_percentage:.2f}")
    print(f"Percentage of companies that set target after {number_of_steps} steps: {target_set_percentage:.2f}")


# # "AverageCommitToTargetTime" column will have the average time at each step for each run
# average_times = data_model["AverageCommitToTargetTime"]

# # You may want to look only at the final averages, which will be the non-null values:
# final_averages = average_times.dropna()


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
#plt.savefig(folder_name + '/aware_comm_target_' + str(num_companies) +'_' + str(number_of_steps) + '.png')
plt.show()




#%% Individual agent- Checking the development of connection number


agent_x_data = data_agent.xs(4, level="AgentID")

# Plot the number of connections over time
plt.plot(agent_x_data.index, agent_x_data["Connections"])
plt.xlabel('Step')
plt.ylabel('Number of Connections')
plt.title('Number of Connections for Agent x over Time')
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
#plt.savefig(folder_name + '/aware_comm_target_sector_breakdown' + str(num_companies) +'_' + str(number_of_steps) + '.png')
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



# # Count the number of companies in each state, grouped by country
# aware_counts = final_agent_data.groupby("country")["aware"].sum()
# committed_counts = final_agent_data.groupby("country")["committed"].sum()
# set_target_counts = final_agent_data.groupby("country")["set_target"].sum()

# total = final_agent_data.groupby("country").size()

# # Merge the data into a single DataFrame
# df_country = pd.concat([total, aware_counts, committed_counts, set_target_counts], axis=1)
# df_country.columns = ["Total","Aware", "Committed", "Set Target"]

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
    #plt.savefig(folder_name + '/aware_comm_target_culturedimensions' + str(num_companies) +'_' + str(number_of_steps) + '.png')
    plt.show()

# Create a grouped bar chart
grouped_bar_chart(
    my_labels, 
    average_culture_dimensions_aware.values, 
    average_culture_dimensions_committed.values,
    average_culture_dimensions_target.values, 
    'Comparison of Average Cultural Dimensions'
)





#%%




failed_agents = model.get_agents_failed_to_set_target()

# convert to a dataframe if desired
failed_agents_df = pd.DataFrame(failed_agents.values(), index=failed_agents.keys())

print(failed_agents_df)

# Count the number of companies per country
country_counts = failed_agents_df['country'].value_counts()

# Plot a bar chart
country_counts.plot(kind='bar', figsize=(10, 6))

# (Optional) Configurations for the plot
plt.title('Number of Companies per Country')
plt.xlabel('Country')
plt.ylabel('Number of Companies')

plt.show()

#%%
from collections import defaultdict

# Initialize a dictionary to hold the sums
country_sums = defaultdict(int)

# Run the simulation multiple times
for _ in range(10):
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
    
    print('works')
    # Run the model
    for _ in range(number_of_steps):
        model.step(shareholder_pressure_lever,
                   manager_pressure_lever,
                   employee_pressure_lever,
                   market_pressure_lever, connectivity_lever)
    
    # Get the failed agents
    failed_agents = model.get_agents_failed_to_set_target()

    # Convert to a dataframe
    failed_agents_df = pd.DataFrame(failed_agents.values(), index=failed_agents.keys())

    # Get the counts per country
    country_counts = failed_agents_df['country'].value_counts()

    # Add these counts to the appropriate entry in the sums dictionary
    for country, count in country_counts.items():
        country_sums[country] += count



# Get the averages by dividing by the number of runs
country_averages = {country: sum_ / 10 for country, sum_ in country_sums.items()}

# Convert to a dataframe for easier plotting
country_averages_df = pd.DataFrame(list(country_averages.items()), columns=['Country', 'Average'])

# Plot the bar chart
country_averages_df.plot(x='Country', y='Average', kind='bar', figsize=(10, 6))
plt.title('Average Number of Failed Companies per Country')
plt.xlabel('Country')
plt.ylabel('Average Number of Failed Companies')
plt.show()


#country_averages_df.to_csv('data_collected/failed_country_averages.csv', index='Country')
#country_averages_df.set_index('Country', inplace=True)



#%%

country_averages_df = pd.read_csv("data_collected/failed_country_averages.csv")
country_averages_df.set_index('Country', inplace=True)




# Specify the path of the Excel file and the name of the worksheet
excel_file_path = 'files/StructuredData.xlsx'

country_df = pd.read_excel(excel_file_path, sheet_name='CountriesUsed', header=0, index_col='Country', nrows = 49) 
scheduling = country_df[['Scheduling']].iloc[0:49]
deciding = country_df[['Deciding']].iloc[0:49]
number_per_country = country_df[['Number']].iloc[0:49]



# Add missing countries to country_averages_df with a default average of 0
for country in number_per_country.index:
    if country not in country_averages_df.index:
        country_averages_df.loc[country] = 0


# Compute the percentage
percentage_df = (country_averages_df['Average'] / number_per_country['Number']) * 100

# Remove the zeros
percentage_df = percentage_df.loc[percentage_df != 0]

# Sort the values in descending order
percentage_df = percentage_df.sort_values(ascending=False)


# Plot the data
percentage_df.plot(kind='bar', figsize=(15,10))
plt.ylabel('Percentage of Failed Companies %', fontsize = 14)
plt.xlabel('Country', fontsize = 14)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("report/base_scenario_failed_percentage.png", dpi = 300, bbox_inches='tight')
plt.show()

#%%
# List of countries with failed companies
failed_countries = percentage_df.index.tolist()

# Filter the dataframes for scheduling and deciding values
scheduling_failed = scheduling.loc[scheduling.index.isin(failed_countries)]
deciding_failed = deciding.loc[deciding.index.isin(failed_countries)]

scheduling_failed = scheduling_failed.reindex(percentage_df.index)
deciding_failed = deciding_failed.reindex(percentage_df.index)

scheduling_failed.loc['Average'] = 54
deciding_failed.loc['Average'] = 61


fig, ax = plt.subplots(figsize=(15,10))

colors = ['blue'] * (len(scheduling_failed) - 1) + ['green']
for i, (idx, row) in enumerate(scheduling_failed.iterrows()):
    ax.bar(i, row['Scheduling'], color=colors[i])

ax.set_xticks(range(len(scheduling_failed)))
ax.set_xticklabels(scheduling_failed.index, rotation=45, fontsize = 14)
plt.ylabel('Scheduling Score', fontsize = 14)
plt.xlabel('Country', fontsize = 14)
plt.yticks(fontsize=14)
#plt.title('Scheduling Scores for Failed Companies per Country')
plt.savefig("report/base_scenario_failed_scheduling.png", dpi = 300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(15,10))
for i, (idx, row) in enumerate(deciding_failed.iterrows()):
    ax.bar(i, row['Deciding'], color=colors[i])

ax.set_xticks(range(len(deciding_failed)))
ax.set_xticklabels(deciding_failed.index, rotation=45, fontsize = 14)
plt.ylabel('Deciding Score', fontsize = 14)
plt.xlabel('Country', fontsize = 14)
plt.yticks(fontsize=14)
#plt.title('Deciding Scores for Failed Companies per Country',fontsize = 12)
plt.savefig("report/base_scenario_failed_deciding.png", dpi = 300, bbox_inches='tight')
plt.show()


