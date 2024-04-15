#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:02:55 2023

@author: charistheodorou

SBTi Model presents the overall model. There are two versions of this due to the changes made 
to the code's mechanisms for the second set of scenarios. This is discussed in Experiments 
and Results Chapter
"""

import mesa
from mesa.time import RandomActivation   #, BaseScheduler
#from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from CompanyAgent_scenarios_set2 import CompanyAgent





#%% SBTi model

class SBTiModel(mesa.Model):

    def __init__(self, num_companies= 100, max_steps= 72,
                 
                 # levers for experiments
                 leadership_lever = 1, 
                 risk_awareness_lever = 1, 
                 reputation_lever = 1, 
                 shareholder_pressure_lever = 1, 
                 manager_pressure_lever =1, 
                 employee_pressure_lever= 1, 
                 market_pressure_lever = 1,  
                 
                 connectivity_lever = 1,
                 market_pressure_on = True,
                 meetings_per_year_scenario = 10,
                 max_comm_duration_scenario = 24,
                 
                 
                 # network
                 alpha = 0.1,
                 beta = 0.1,
                 gamma = 0.1,
                 delta = 0.01,
                 rewiring_frequency = 13,
                 
                 #awareness
                 aware_update_step = 4,               # aware due to interaction with other companies
                 companies_turn_aware_per_round = 1,  # how often companies turn aware due to SBTi campaigns
                 steps_per_round = 6,  
                 
                 #Committing
                 meetings_per_year = 10,    # average of existing data- used to check if a company will commit
                 pres_mot_eval = "product", # serial, sum, product
                 
                 manufacturing_coefficient = 0.232,
                 non_manufacturing_coefficient = 0.358,
                 shareholder_pressure_coefficient = 0.253,
                 manager_pressure_coefficient = 0.254,
                 employee_pressure_coefficient = 0.238,
                 market_pressure_coefficient = 0.210,
                 
                 
                 pressure_threshold = 5,
                 motivation_threshold = 5,
                 
                 
                 
                 #setting a target
                 work_rate = 1/6,
                 internal_target_range=[0, 1],
                 max_comm_duration = 24,
                 

                 
                 
                 sigma=0.025, seed=None):
        
        super().__init__(seed=seed)
        
        #%% Import data

        # Specify the path of the Excel file and the name of the worksheet
        excel_file_path = 'files/StructuredData.xlsx'
        
        # Read the data from the worksheet into a dataframe
        sector_df = pd.read_excel(excel_file_path, sheet_name='SectorsUsed', header=0, index_col='Sector', nrows = 8)
        country_df = pd.read_excel(excel_file_path, sheet_name='CountriesUsed', header=0, index_col='Country', nrows = 49)

#%% Data as input to the model

        #COUNTRY DATA
        self.country_probs = country_df[['Percentage']].iloc[0:49]


        self.mu_communicating = country_df[['Communicating']].iloc[0:49]/100
        self.mu_evaluating = country_df[['Evaluating']].iloc[0:49]/100
        self.mu_leading = country_df[['Leading']].iloc[0:49]/100
        self.mu_deciding = country_df[['Deciding']].iloc[0:49]/100
        self. mu_trusting = country_df[['Trusting']].iloc[0:49]/100
        self.mu_disagreeing = country_df[['Disagreeing']].iloc[0:49]/100
        self.mu_scheduling = country_df[['Scheduling']].iloc[0:49]/100
        #mu_persuading = country_df[['Persuading']].iloc[0:50]


        # SECTOR DATA
        self.sector_probs = sector_df[['Percentage']].iloc[0:13]
        self.emissions = sector_df[['Emissions per company (Mt)']].iloc[0:13]
        self.sector_type = sector_df[['Sector Type']].iloc[0:13]
        
        
        self.num_companies = num_companies
        self.run_time = max_steps
#        self.average_commit_to_target_time = None
        

        # awareness
        self.companies_turn_aware_per_round = companies_turn_aware_per_round
        self.steps_per_round = steps_per_round
        
        
        # commitment
        self.information = self.random.random()
        self.communication = self.random.random()
        self.monitoring = self.random.random()
        self.benefits = self.random.random()
        
        self.sbti_attributes = (self.information + self.communication 
                                + self.monitoring + self.benefits)
        
        self.shareholder_pressure_coefficient = shareholder_pressure_coefficient
        self.manager_pressure_coefficient = manager_pressure_coefficient
        self.employee_pressure_coefficient = employee_pressure_coefficient
        self.market_pressure_coefficient = market_pressure_coefficient
        
        self.pressure_threshold = pressure_threshold
        self.motivation_threshold = motivation_threshold
        self.max_comm_duration = max_comm_duration
        
        # network
        self.schedule = RandomActivation(self)
        self.grid = mesa.space.MultiGrid(20, 20, True)
        
        # define weights for connectivity matrix
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.agents = []
        
        
        #Gaussian Distribution
        #self.sigma = sigma
        self.initial_agent_data = []
        
        self.rewiring_frequency = rewiring_frequency
        # Recording steps
        self.step_counter = 1
        

        
        # add nodes for companies to network
        for i in range(self.num_companies):
             a = CompanyAgent(i, self,
                              # awareness
                              aware_update_step,
                              
                              # committing
                              meetings_per_year,
                              pres_mot_eval,
                              manufacturing_coefficient,
                              non_manufacturing_coefficient,
                              
                              #setting a target
                              work_rate,
                              internal_target_range,
                              
                              sigma)
             
             self.schedule.add(a)
             self.agents.append(a)
             
             # Add the agent to a random grid cell
             x = self.random.randrange(self.grid.width)
             y = self.random.randrange(self.grid.height)
             self.grid.place_agent(a, (x, y))
            
            

        # create initial network based on connectivity matrix M
        self.M = self.generateM()
        self.G = self.generateNetwork()
        
        
        
        #campaigns

        
        self.leadership_lever =             1
        self.risk_awareness_lever =         1
        self.reputation_lever =             1
        

        self.shareholder_pressure_lever =   1
        self.manager_pressure_lever =       1
        self.employee_pressure_lever =      1
        self.market_pressure_lever =        1
        self.market_pressure_on = market_pressure_on
        
        self.connectivity_lever = 1
        #self.print_agent_connections()
        
        
        # Initialize the NetworkModule to visualize agents and connections
        #self.network =  NetworkModule(self.G, canvas_height=500, canvas_width=800)

        # seed for random number generation
        if seed:
#            random.seed(seed)
            np.random.seed(seed)
        

                    
        # Add the DataCollector to your model
        self.datacollector = DataCollector(model_reporters={
            "NotAware": lambda m: sum([not agent.is_aware for agent in m.schedule.agents]),
            "Aware": lambda m: sum([agent.is_aware for agent in m.schedule.agents]),
            "Committed": lambda m: sum([agent.is_committed for agent in m.schedule.agents]),
            "HasTarget": lambda m: sum([agent.has_target for agent in m.schedule.agents]),
            
            "TotalBySector": lambda m: m.total_companies_by_sector(), 
            "AwareBySector": lambda m: m.get_aware_by_sector(),
            "CommittedBySector": lambda m: m.get_committed_by_sector(),
            "SetTargetBySector": lambda m: m.get_target_set_by_sector(),
            
            "TotalByCountry": lambda m: m.total_companies_by_country(), 
            "AwareByCountry": lambda m: m.get_aware_by_country(),
            "CommittedByCountry": lambda m: m.get_committed_by_country(),
            "SetTargetByCountry": lambda m: m.get_target_set_by_country(),
            
            "AverageCommunicatingAware": lambda m: m.get_average_communicating_aware(),
            "AverageEvaluatingAware": lambda m: m.get_average_evaluating_aware(),
            "AverageLeadingAware": lambda m: m.get_average_leading_aware(),
            "AverageDecidingAware": lambda m: m.get_average_deciding_aware(),
            "AverageTrustingAware": lambda m: m.get_average_trusting_aware(),
            "AverageDisagreeingAware": lambda m: m.get_average_disagreeing_aware(),
            "AverageSchedulingAware": lambda m: m.get_average_scheduling_aware(),
            
            "AverageCommunicatingCommitted": lambda m: m.get_average_communicating_committed(),
            "AverageEvaluatingCommitted": lambda m: m.get_average_evaluating_committed(),
            "AverageLeadingCommitted": lambda m: m.get_average_leading_committed(),
            "AverageDecidingCommitted": lambda m: m.get_average_deciding_committed(),
            "AverageTrustingCommitted": lambda m: m.get_average_trusting_committed(),
            "AverageDisagreeingCommitted": lambda m: m.get_average_disagreeing_committed(),
            "AverageSchedulingCommitted": lambda m: m.get_average_scheduling_committed(),
            
            "AverageCommunicatingSetTarget": lambda m: m.get_average_communicating_set_target(),
            "AverageEvaluatingSetTarget": lambda m: m.get_average_evaluating_set_target(),
            "AverageLeadingSetTarget": lambda m: m.get_average_leading_set_target(),
            "AverageDecidingSetTarget": lambda m: m.get_average_deciding_set_target(),
            "AverageTrustingSetTarget": lambda m: m.get_average_trusting_set_target(),
            "AverageDisagreeingSetTarget": lambda m: m.get_average_disagreeing_set_target(),
            "AverageSchedulingSetTarget": lambda m: m.get_average_scheduling_set_target(),
            
            
            "AverageCommitToTargetTime": lambda m: m.average_time_to_set_target(),  
            "AverageConnections": SBTiModel.get_average_connections,},
        
            agent_reporters={
        #     "committed": lambda a: a.is_committed,
        #     "aware": lambda a: a.is_aware,
        #     "set_target": lambda a: a.has_target,
        #     "sector": lambda a: a.sector,
        #     "country": lambda a: a.country,
           "Connections": lambda a: len(list(a.model.G.neighbors(a))) if a.model.G.has_node(a) else 0,}
        #     "Motivation": "motivation", 
        #     "Pressure": "pressure",
        #     "Communicating": "communicating", 
        #     "Evaluating": "evaluating", 
        #     "Leading": "leading",
        #     "Deciding": "deciding", 
        #     "Trusting": "trusting", 
        #     "Disagreeing": "disagreeing",
        #     "Scheduling": "scheduling"}
    )
    
    
    
    # create connectivity matrix M
    def generateM(self):
        M = np.zeros((self.num_companies, self.num_companies))
        for i in range(self.num_companies):
            for j in range(self.num_companies):
                if i == j:
                    M[i, j] = 0  # no self-loops
                    
                else:
                    M[i, j] += self.delta
                    if self.schedule.agents[i].sector == self.schedule.agents[j].sector:
                        M[i, j] += self.alpha
                    if self.schedule.agents[i].country == self.schedule.agents[j].country:
                        M[i, j] += self.beta
                        
                        # add similarity based on communicating attribute
                    communicating_difference = self.schedule.agents[i].communicating - self.schedule.agents[j].communicating
                    communicating_similarity = np.exp(-0.2 * communicating_difference**2)
                    M[i, j] += communicating_similarity*self.gamma
        
        return M
    
    
    
    
    
    def generateNetwork(self):
        # create network based on connectivity matrix M
        G = nx.DiGraph()
        
        for i in range(self.num_companies):
            for j in range(self.num_companies):
                if i != j and self.random.random() < self.M[i][j]:
                    G.add_edge(self.agents[i], self.agents[j])
        return G
    
    
    #def print_agent_connections(self):
    #    for agent in self.agents:
    #        if self.G.has_node(agent):
    #            num_connections = len(list(self.G.neighbors(agent)))
    #        else:
    #            num_connections = 0
    #        print(f"Agent {agent.unique_id} has {num_connections} connections.")
    


         
    def visualize_network(self):
        """Create a complete graph with all agents as nodes"""
        all_agents = self.agents
        complete_graph = nx.DiGraph()
        complete_graph.add_nodes_from(all_agents)
        
        # Add the edges based on the actual connections in your model
        complete_graph.add_edges_from(self.G.edges())
        
        pos = nx.spring_layout(complete_graph)
        
        # Create a dictionary mapping agents to their unique_id
        labels = {agent: agent.unique_id for agent in all_agents}
        
        nx.draw(complete_graph, pos, labels=labels, node_color="red", with_labels=True, node_size=200)
        nx.draw_networkx_edge_labels(complete_graph, pos)
        plt.show()

    def collect_initial_connections(self):
        """Collect initial number of connections for each agent after the first step."""
        self.initial_connections_data = []
        for agent in self.schedule.agents:
            agent.update_neighbors()
            agent_connections = {
                'unique_id': agent.unique_id,
                'initial_connections': len(agent.neighbors)
            }
            self.initial_connections_data.append(agent_connections)
    
    


    
    
    def campaign_effect(self, shareholder_pressure_lever,
                        manager_pressure_lever,
                        employee_pressure_lever,
                        market_pressure_lever, connectivity_lever, meetings_per_year_scenario, max_comm_duration_scenario):
         "Applies the campaign effect on shareholder pressure."
         if self.step_counter == 64:
             # 56% reported direct influence in 2020
             self.shareholder_pressure_lever = shareholder_pressure_lever
             # Make all agents aware during the campaign
             for agent in self.schedule.agents:
                agent.is_aware = True
         # elif self.step_counter == 88:
         #     # 1/3 reported direct influence in 2021 and onwards
             self.manager_pressure_lever =       manager_pressure_lever
             self.employee_pressure_lever =      employee_pressure_lever
             self.market_pressure_lever =        market_pressure_lever
             
             self.connectivity_lever = connectivity_lever
             self.max_comm_duration_scenario = max_comm_duration_scenario
             self.meetings_per_year = meetings_per_year_scenario
             
             
         
    def get_agents_failed_to_set_target(self):
        failed_agents = {}
        for agent in self.schedule.agents:
            if agent.failed_to_set_target:
                agent_data = {
                    "country": agent.country,
                    "sector": agent.sector,
                    "Communicating": agent.communicating, 
                    "Evaluating": agent.evaluating, 
                    "Leading": agent.leading,
                    "Deciding": agent.deciding, 
                    "Trusting": agent.trusting, 
                    "Disagreeing": agent.disagreeing,
                    "Scheduling": agent.scheduling
                }
                failed_agents[agent.unique_id] = agent_data
        return failed_agents     
   
    
   
    def step(self, shareholder_pressure_lever,
             manager_pressure_lever,
             employee_pressure_lever,
             market_pressure_lever, connectivity_lever, meetings_per_year_scenario,max_comm_duration_scenario):
        
        self.datacollector.collect(self)
        self.schedule.step()
        
        #if self.step_counter == 1:
        #    self.collect_initial_connections()
        
        # Apply campaign effects
        self.campaign_effect(shareholder_pressure_lever,
                             manager_pressure_lever,
                             employee_pressure_lever,
                             market_pressure_lever, 
                             connectivity_lever, meetings_per_year_scenario,max_comm_duration_scenario)
        
        
        # Rewire all agents
        if self.schedule.steps % (self.rewiring_frequency / self.connectivity_lever) == 0:
            #print(self.step_counter)
            for agent in self.schedule.agents:
                agent.rewire()
                    
        # Select agents randomly and make them aware
        if self.schedule.steps % self.steps_per_round == 0:
            num_companies_to_awake = self.companies_turn_aware_per_round
            companies_to_awake = self.random.sample(self.schedule.agents, num_companies_to_awake)
            for agent in companies_to_awake:
                agent.is_aware = True
        
        self.step_counter += 1
        
        # When reaching the maximum number of steps, finalize and calculate end-of-run statistics
        #if self.step_counter == self.run_time:
        #    self.finalize()
    
    
#    def finalize(self):
#        self.average_commit_to_target_time = self.average_time_to_set_target()
    
    def average_time_to_set_target(self):
        total_time = 0
        count = 0
        for agent in self.schedule.agents:
            if agent.commit_step is not None and agent.target_set_step is not None:
                total_time += agent.target_set_step - agent.commit_step
                count += 1
        if count > 0:
            return total_time / count
        else:
            return np.nan


    def get_aware_total(self):
        return sum([1 for agent in self.schedule.agents if agent.is_aware])
    
    def get_aware_total_percent(self):
        return self.get_aware_total() / self.num_companies

    def get_committed_total(self):
        return sum([1 for agent in self.schedule.agents if agent.is_committed])
    
    def get_committed_total_percent(self):
        return self.get_committed_total() / self.num_companies

    def get_target_set_total(self):
        return sum([1 for agent in self.schedule.agents if agent.has_target])
    
    def get_target_set_total_percent(self):
        return self.get_target_set_total() / self.num_companies
    
    def total_companies_by_country(self):
        result = {}
        for country in self.country_probs.index:
            result[country] = sum([1 for agent in self.schedule.agents if agent.country == country])
        return result
    
    def total_companies_by_sector(self):
        result = {}
        for sector in self.sector_probs.index:
            result[sector] = sum([1 for agent in self.schedule.agents if agent.sector == sector])
        return result
    
    def get_aware_by_country(self):
        result = {}
        for country in self.country_probs.index:
            result[country] = sum([1 for agent in self.schedule.agents if agent.is_aware and agent.country == country])
        return result

    def get_aware_by_sector(self):
        result = {}
        for sector in self.sector_probs.index:
            result[sector] = sum([1 for agent in self.schedule.agents if agent.is_aware and agent.sector == sector])
        return result
    
    def get_committed_by_country(self):
        result = {}
        for country in self.country_probs.index:
            result[country] = sum([1 for agent in self.schedule.agents if agent.is_committed and agent.country == country])
        return result
    
    def get_committed_by_sector(self):
        result = {}
        for sector in self.sector_probs.index:
            result[sector] = sum([1 for agent in self.schedule.agents if agent.is_committed and agent.sector == sector])
        return result
    
    def get_target_set_by_country(self):
        result = {}
        for country in self.country_probs.index:
            result[country] = sum([1 for agent in self.schedule.agents if agent.has_target and agent.country == country])
        return result
    
    def get_target_set_by_sector(self):
        result = {}
        for sector in self.sector_probs.index:
            result[sector] = sum([1 for agent in self.schedule.agents if agent.has_target and agent.sector == sector])
        return result
    
    def aware_count(self):
        aware_count = sum([1 for agent in self.schedule.agents if agent.is_aware])
        return aware_count
    
    def committed_count(self):
        committed_count = sum([1 for agent in self.schedule.agents if agent.is_committed])
        return committed_count
    
    def target_set_count(self):
        target_set_count = sum([1 for agent in self.schedule.agents if agent.has_target])
        return target_set_count
#%% Culture dimensions data collectio    


    
    # For aware agents
    def get_average_communicating_aware(self):
        communicating_sum = sum([agent.communicating for agent in self.schedule.agents if agent.is_aware])
        aware_count = sum([1 for agent in self.schedule.agents if agent.is_aware])
        return communicating_sum / aware_count if aware_count != 0 else 0
    
    def get_average_evaluating_aware(self):
        evaluating_sum = sum([agent.evaluating for agent in self.schedule.agents if agent.is_aware])
        aware_count = sum([1 for agent in self.schedule.agents if agent.is_aware])
        return evaluating_sum / aware_count if aware_count != 0 else 0    
    
    def get_average_leading_aware(self):
        leading_sum = sum([agent.leading for agent in self.schedule.agents if agent.is_aware])
        aware_count = sum([1 for agent in self.schedule.agents if agent.is_aware])
        return leading_sum / aware_count if aware_count != 0 else 0
    
    def get_average_deciding_aware(self):
        deciding_sum = sum([agent.deciding for agent in self.schedule.agents if agent.is_aware])
        aware_count = sum([1 for agent in self.schedule.agents if agent.is_aware])
        return deciding_sum / aware_count if aware_count != 0 else 0
    
    def get_average_trusting_aware(self):
        trusting_sum = sum([agent.trusting for agent in self.schedule.agents if agent.is_aware])
        aware_count = sum([1 for agent in self.schedule.agents if agent.is_aware])
        return trusting_sum / aware_count if aware_count != 0 else 0
    
    def get_average_disagreeing_aware(self):
        disagreeing_sum = sum([agent.disagreeing for agent in self.schedule.agents if agent.is_aware])
        aware_count = sum([1 for agent in self.schedule.agents if agent.is_aware])
        return disagreeing_sum / aware_count if aware_count != 0 else 0
    
    def get_average_scheduling_aware(self):
        scheduling_sum = sum([agent.scheduling for agent in self.schedule.agents if agent.is_aware])
        aware_count = sum([1 for agent in self.schedule.agents if agent.is_aware])
        return scheduling_sum / aware_count if aware_count != 0 else 0
    
    # Repeat the above for committed agents
    def get_average_communicating_committed(self):
        communicating_sum = sum([agent.communicating for agent in self.schedule.agents if agent.is_committed])
        committed_count = sum([1 for agent in self.schedule.agents if agent.is_committed])
        return communicating_sum / committed_count if committed_count != 0 else 0
    
    def get_average_evaluating_committed(self):
        evaluating_sum = sum([agent.evaluating for agent in self.schedule.agents if agent.is_committed])
        committed_count = sum([1 for agent in self.schedule.agents if agent.is_committed])
        return evaluating_sum / committed_count if committed_count != 0 else 0
    
    def get_average_leading_committed(self):
        leading_sum = sum([agent.leading for agent in self.schedule.agents if agent.is_committed])
        committed_count = sum([1 for agent in self.schedule.agents if agent.is_committed])
        return leading_sum / committed_count if committed_count != 0 else 0
    
    def get_average_deciding_committed(self):
        deciding_sum = sum([agent.deciding for agent in self.schedule.agents if agent.is_committed])
        committed_count = sum([1 for agent in self.schedule.agents if agent.is_committed])
        return deciding_sum / committed_count if committed_count != 0 else 0
    
    def get_average_trusting_committed(self):
        trusting_sum = sum([agent.trusting for agent in self.schedule.agents if agent.is_committed])
        committed_count = sum([1 for agent in self.schedule.agents if agent.is_committed])
        return trusting_sum / committed_count if committed_count != 0 else 0
    
    def get_average_disagreeing_committed(self):
        disagreeing_sum = sum([agent.disagreeing for agent in self.schedule.agents if agent.is_committed])
        committed_count = sum([1 for agent in self.schedule.agents if agent.is_committed])
        return disagreeing_sum / committed_count if committed_count != 0 else 0
    
    def get_average_scheduling_committed(self):
        scheduling_sum = sum([agent.scheduling for agent in self.schedule.agents if agent.is_committed])
        committed_count = sum([1 for agent in self.schedule.agents if agent.is_committed])
        return scheduling_sum / committed_count if committed_count != 0 else 0
    
    # Repeat the above for set_target agents
    def get_average_communicating_set_target(self):
        communicating_sum = sum([agent.communicating for agent in self.schedule.agents if agent.has_target])
        set_target_count = sum([1 for agent in self.schedule.agents if agent.has_target])
        return communicating_sum / set_target_count if set_target_count != 0 else 0

    def get_average_evaluating_set_target(self):
        evaluating_sum = sum([agent.evaluating for agent in self.schedule.agents if agent.has_target])
        set_target_count = sum([1 for agent in self.schedule.agents if agent.has_target])
        return evaluating_sum / set_target_count if set_target_count != 0 else 0
    
    def get_average_leading_set_target(self):
        leading_sum = sum([agent.leading for agent in self.schedule.agents if agent.has_target])
        set_target_count = sum([1 for agent in self.schedule.agents if agent.has_target])
        return leading_sum / set_target_count if set_target_count != 0 else 0
    
    def get_average_deciding_set_target(self):
        deciding_sum = sum([agent.deciding for agent in self.schedule.agents if agent.has_target])
        set_target_count = sum([1 for agent in self.schedule.agents if agent.has_target])
        return deciding_sum / set_target_count if set_target_count != 0 else 0
    
    def get_average_trusting_set_target(self):
        trusting_sum = sum([agent.trusting for agent in self.schedule.agents if agent.has_target])
        set_target_count = sum([1 for agent in self.schedule.agents if agent.has_target])
        return trusting_sum / set_target_count if set_target_count != 0 else 0
    
    def get_average_disagreeing_set_target(self):
        disagreeing_sum = sum([agent.disagreeing for agent in self.schedule.agents if agent.has_target])
        set_target_count = sum([1 for agent in self.schedule.agents if agent.has_target])
        return disagreeing_sum / set_target_count if set_target_count != 0 else 0
    
    def get_average_scheduling_set_target(self):
        scheduling_sum = sum([agent.scheduling for agent in self.schedule.agents if agent.has_target])
        set_target_count = sum([1 for agent in self.schedule.agents if agent.has_target])
        return scheduling_sum / set_target_count if set_target_count != 0 else 0
    
        
    def get_average_connections(self):
        total_connections = sum([agent.num_connections for agent in self.schedule.agents])
        average_connections = total_connections / len(self.schedule.agents)
        return average_connections
    
