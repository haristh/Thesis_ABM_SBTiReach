#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:33:49 2023

@author: charistheodorou

One of the two main files that represent ABM model alongside SBTiModel (both versions). 
It was implemented based on the Conceptualisation and Formalisation chapter.
"""
import mesa
from mesa.datacollection import DataCollector
import numpy as np



class CompanyAgent(mesa.Agent):
    """An agent representing a company that joins SBTi."""

    def __init__(self, unique_id, model, 
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
                              
                              sigma):
        
        super().__init__(unique_id, model)
        
        
        
        # PARAMETERS
        #1. Country-based parameters
        self.country = self.choose_country(self.model.country_probs) # probabilities based on SBTi progress report
        
        # Culture dimension values assigned
        self.communicating =    np.clip(self.model.random.gauss(self.model.mu_communicating.loc[self.country].values[0],sigma),0,1)
        self.evaluating =       np.clip(self.model.random.gauss(self.model.mu_evaluating.loc[self.country].values[0],sigma),0,1)
        self.leading =          np.clip(self.model.random.gauss(self.model.mu_leading.loc[self.country].values[0],sigma),0,1)
        self.deciding =         np.clip(self.model.random.gauss(self.model.mu_deciding.loc[self.country].values[0],sigma),0,1)
        self.trusting =         np.clip(self.model.random.gauss(self.model.mu_trusting.loc[self.country].values[0],sigma),0,1)
        self.disagreeing =      np.clip(self.model.random.gauss(self.model.mu_disagreeing.loc[self.country].values[0],sigma),0,1)
        self.scheduling =       np.clip(self.model.random.gauss(self.model.mu_scheduling.loc[self.country].values[0],sigma),0,1)




        #2. Sector-based paremeters
        self.sector = self.choose_sector(self.model.sector_probs) # probabilities based on SBTi progress report
        self.emissions = self.model.random.gauss(self.model.emissions.loc[self.sector].
                                                 values[0],sigma) # emissions based on CDP Europe
        
        self.sector_type = self.model.sector_type.loc[self.sector].values[0]
        
        #3. others
        #self.internal_target_coefficient = 0.25
        #self.internal_target = self.model.random.random()  #play with it for parameter tuning
        self.internal_target_min, self.internal_target_max = internal_target_range
        self.internal_target= (self.model.random.random() * (self.internal_target_max - self.internal_target_min)
                                   + self.internal_target_min)
        
        # VARIABLES
        
        self.neighbors = None           # initialized after the initialization of the model network
        self.num_connections = 0     # initialized after the initialization of the model network
        
        #processes
        #awareness
        self.aware_update_step = aware_update_step
        #commitment
        self.pressure = None
        self.motivation = None
        self.pres_mot_eval= pres_mot_eval
        self.meetings_per_year= meetings_per_year
        # Add sector factor
        if self.sector_type == "Manufacturing":
            self.sector_factor = manufacturing_coefficient
        elif self.sector_type == "Non-manufacturing":
            self.sector_factor = non_manufacturing_coefficient
        else: # both
            self.sector_factor = (manufacturing_coefficient + non_manufacturing_coefficient)/2
        
        
        
        
        #setting target
        self.target_progress = self.internal_target
        self.commitment_duration = 0
        self.work_rate = work_rate
        self.failed_to_set_target = False # initialize the attribute
        
        #states of agent
        self.is_aware = False
        self.is_committed = False
        self.has_target = False

        
        # Generate a random number from a uniform distribution in the range [0, 1)
        
        self.leadership = self.model.random.random()
        self.riskawareness= self.model.random.random()
        self.reputation= self.model.random.random()
        
        
            
        # Recording steps
        self.step_counter = 1
        self.commit_step = None
        self.target_set_step = None
        
        # Save initial parameters
        agent_params = {
        'unique_id': self.unique_id,
        'Country': self.country,
        'Sector': self.sector,
        'Emissions':self.emissions,
        'Type': self.sector_type,
        'Internal Target': self.internal_target,
        'Communicating':self.communicating,
        'Evaluating': self.evaluating, 
        'Leading': self.leading,
        'Decidng': self.deciding,
        'Trusting': self.trusting,
        'Disagreeing': self.disagreeing,
        'Scheduling': self.scheduling,
        'Climate Leadership': self.leadership,
        'Climate risks': self.riskawareness,
        'Climate Reputation':self.reputation,
        }
        
        self.model.initial_agent_data.append(agent_params)
    

        
        
    def choose_country(self, country_probs):
        """Selects a country based on the given probabilities."""
        countries = country_probs.index.tolist()
        probs = country_probs.Percentage.tolist()
        return self.model.random.choices(countries, probs)[0]

    def choose_sector(self, sector_probs):
        """Selects a sector based on the given probabilities."""
        sectors = sector_probs.index.tolist()
        probs = sector_probs.Percentage.tolist()
        return self.model.random.choices(sectors, probs)[0]
    
    def update_neighbors(self):
        """Update the list of this agent's neighbors in the network and return it."""
        self.neighbors = list(self.model.G.neighbors(self)) if self.model.G.has_node(self) else []
    
    def commit_to_sbti(self):
        "Calculates pressure and motivation for each agent."
        
        #Pressure
        # Internal pressure
        shareholder_pressure = (1 - self.disagreeing)  * self.model.shareholder_pressure_coefficient * self.model.shareholder_pressure_lever

        manager_pressure = (1- self.scheduling) * self.model.manager_pressure_coefficient * self.model.manager_pressure_lever
        employee_pressure = (1 - self.leading) * self.model.employee_pressure_coefficient * self.model.employee_pressure_lever
        internal_pressure = (shareholder_pressure + manager_pressure + employee_pressure)/3
        
        # Market pressure
        if self in self.model.G:
            # All neighbors
            neighbors = list(self.model.G.neighbors(self))
        
            # Neighbors that have committed
            committed_neighbors = [neighbor for neighbor in neighbors if neighbor.is_committed]

            if len(neighbors) > 0: 
                market_pressure = self.trusting * len(committed_neighbors)/len(neighbors) * self.model.market_pressure_coefficient *self.model.market_pressure_lever
               # print('committed:',len(committed_neighbors),'total:', len(neighbors))
            else:
                market_pressure = 0
                    
        else:
            market_pressure = 0
        
        self.pressure = (internal_pressure + market_pressure) * self.sector_factor
        

        # Motivation based on organisational sbti attributes and corporate motivation - Fink
        # individual motivations
        self.motivation  = ((self.riskawareness * self.disagreeing * self.model.risk_awareness_lever) 
                            + (self.reputation * self.trusting * self.model.reputation_lever) 
                            + (self.leadership * self.leading *self.model.leadership_lever))
        total = self.motivation + self.model.sbti_attributes

        

        
        #Thresholds
        random_number_pressure = self.model.random.random()* self.model.pressure_threshold
        random_number_motivation = self.model.random.random()* self.model.motivation_threshold
        
        if self.pres_mot_eval =="sum":
            threshold_sum = random_number_pressure + random_number_motivation
            
            if self.pressure + total > threshold_sum:
                self.is_committed = True # company becomes committed
                self.commit_step = self.model.schedule.steps
            else:
                self.is_committed = False
        
        if self.pres_mot_eval =="product":
            threshold_prod = random_number_pressure * random_number_motivation
            
            if self.pressure * total > threshold_prod:
                self.is_committed = True # company becomes committed
                self.commit_step = self.model.schedule.steps
            else:
                self.is_committed = False
        
        
        if self.pres_mot_eval =="serial":

            if self.pressure > random_number_pressure:
                # individual motivations and sbti attributes compared with a variable
                if total > random_number_motivation:
                    self.is_committed = True # company becomes committed
                    self.commit_step = self.model.schedule.steps
                else:
                    self.is_committed = False
            
            else:
                self.is_committed = False


    def rewire(self):
        "Rewiring the network. Companies get connected with new companies"
        
        # We choose from all agents except self and current neighbors
        current_neighbors = list(self.model.G.neighbors(self)) if self.model.G.has_node(self) else [] 
        potential_new_neighbors = sorted(set(self.model.schedule.agents), key=lambda x: x.unique_id)
        potential_new_neighbors = [agent for agent in potential_new_neighbors if agent not in current_neighbors and agent != self]
        
        if not potential_new_neighbors:
            return
    
        if self.model.market_pressure_on and self.model.step_counter >= 64:
            # Split potential neighbors into committed and not committed
            committed_neighbors = [agent for agent in potential_new_neighbors if agent.is_committed]
            not_committed_neighbors = [agent for agent in potential_new_neighbors if not agent.is_committed]
    
            # Favor committed neighbors with a certain probability, let's say 90%
            if committed_neighbors and self.model.random.random() < 0.9:
                new_neighbor = self.model.random.choice(committed_neighbors)
            else:
                new_neighbor = self.model.random.choice(not_committed_neighbors)
        else:
            new_neighbor = self.model.random.choice(potential_new_neighbors)
    
        # Calculate connection probability
        connect_prob = self.model.M[self.model.schedule.agents.index(self), self.model.schedule.agents.index(new_neighbor)]
        
        # Connect based on probability
        if self.model.random.random() < connect_prob:
            self.model.G.add_edge(self, new_neighbor)
    
            # If agent is not connected to any other agents, then it cannot disconnect
            if not self.model.G.has_node(self):
                return
    
            # List of current neighbors (connected agents)
            neighbors = sorted(list(self.model.G.neighbors(self)), key=lambda x: x.unique_id)
    
            # If agent has no neighbors, it cannot disconnect
            if not neighbors:
                return
    
            # Choose a neighbor to disconnect
            neighbor_to_remove = self.model.random.choice(neighbors)
    
            # Calculate disconnection probability
            disconnect_prob = 1- self.model.M[self.model.schedule.agents.index(self), self.model.schedule.agents.index(neighbor_to_remove)]
    
            # Disconnect based on probability
            if self.model.random.random() < disconnect_prob:
                self.model.G.remove_edge(self, neighbor_to_remove)




    
#   STEP    
    def step(self):
        self.update_neighbors()  # Updates the list of neighbors at the start of each step.
        
        if self.neighbors:  # Will evaluate to True only if self.neighbors is not an empty list.
            self.num_connections = len(self.neighbors)
        
        
        # Updating awareness based on network
        if not self.is_aware and self.num_connections>0:
            
            
            if self.step_counter % self.aware_update_step==0:
                aware_neighbors = [neighbor for neighbor in self.neighbors if neighbor.is_aware]
                
                #probability of becoming aware is assumed to be equal to the fraction of neighbors that are aware at each step.
                prob_become_aware = len(aware_neighbors) / self.num_connections
                
                if self.model.random.random() < prob_become_aware:
                    self.is_aware = True
            
            
        # Updating commitment based on pressure and motivation
        elif self.is_aware and not self.is_committed and self.model.random.random() < self.meetings_per_year/12:
            self.commit_to_sbti()

            
            
        # Updating work towards submission/ setting target
        elif self.is_aware and self.is_committed and not self.has_target:
            # Increase the commitment_duration since the company has not set a target yet
            self.commitment_duration += 1
            
            # If commitment_duration is more than 6 steps, revert is_committed to False
            if self.commitment_duration > self.model.max_comm_duration:
                self.is_committed = False
                self.commitment_duration = 0  # Reset commitment duration
                self.failed_to_set_target = True if not self.has_target else False
            else:
                if self.scheduling < self.model.random.random(): #There was a mistake with the inequality
                    self.target_progress+= self.deciding * self.work_rate
                    
                    # Code to determine if company sets target with SBTi
                    if self.target_progress>1:
                        self.has_target = True
                        self.target_set_step = self.model.schedule.steps


        else:
            # company has awareness, commitment, and target set with SBTi
            pass  # code for companies with awareness, commitment, and target set]
        
        self.step_counter += 1
            
