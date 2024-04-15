#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:06:23 2023

@author: charistheodorou
This files contains the code needed for the Parameter tuning. The most significant parameters are optimised based on the % committed, % set target and average time between commitment and set target desired outcomes.

"""

from ema_workbench import (Model, 
                           RealParameter,
                           CategoricalParameter,
                           IntegerParameter,
                           ScalarOutcome,
                           TimeSeriesOutcome,
                           ArrayOutcome, 
                           ema_logging, 
                           perform_experiments,
                           save_results,
                           load_results,
                           MultiprocessingEvaluator,
                           Constant)
from ema_workbench import Constraint

from ema_workbench.analysis import pairs_plotting, prim

from SALib.analyze import sobol
from ema_workbench import Samplers
from ema_workbench.em_framework.salib_samplers import get_SALib_problem


import matplotlib.pyplot as plt
import statsmodels.api as sm
from SBTiModel_main import SBTiModel

from scipy import stats
import seaborn as sns
sns.set_style('white')
import seaborn as sns
import pandas as pd
import numpy as np


# Set the display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def sbti_model(num_companies= 2233, max_steps= 60,
             
             # levers for experiments
             leadership_lever = 1, 
             risk_awareness_lever = 1, 
             reputation_lever = 1, 
             shareholder_pressure_lever = 1, 
             manager_pressure_lever =1, 
             employee_pressure_lever= 1, 
             market_pressure_lever = 1,  
             
             # network
             alpha = 0.1,
             beta = 0.1,
             gamma = 0.1,
             delta = 0.01,
             rewiring_frequency = 1,
             
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

    
    
    # Create a new instance of the model with the given parameters
    model = SBTiModel( num_companies= num_companies, 
            max_steps= max_steps,
                     
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
    
    
    aware_count_yearly = []
    committed_count_yearly = []
    target_set_count_yearly = []
    
    for i in range(max_steps):
        model.step(shareholder_pressure_lever,
                 manager_pressure_lever,
                 employee_pressure_lever,
                 market_pressure_lever)
        if i % 12 == 0:  # Collect yearly data
            aware_count_yearly.append(model.aware_count())
            committed_count_yearly.append(model.committed_count())
            target_set_count_yearly.append(model.target_set_count())

    final_aware_total_percent = model.get_aware_total_percent()
    final_committed_total_percent = model.get_committed_total_percent()
    final_target_set_total_percent = model.get_target_set_total_percent()

    
    final_aware_by_country = model.get_aware_by_country()
    final_aware_by_sector = model.get_aware_by_sector()
    
    final_committed_by_country = model.get_committed_by_country()
    final_committed_by_sector = model.get_committed_by_sector()
    
    final_target_set_by_country = model.get_target_set_by_country()
    final_target_set_by_sector = model.get_target_set_by_sector()
    
    
    
    average_time_to_set_target = model.average_time_to_set_target()
    
    
    target_final_committed_total_percent = 0.20
    target_final_target_set_total_percent = 0.12
    target_average_time_to_set_target  = 6.4
    
    # Calculate absolute differences from target values
    final_committed_difference = abs(final_committed_total_percent - target_final_committed_total_percent)
    final_target_set_difference = abs(final_target_set_total_percent - target_final_target_set_total_percent)
    average_time_to_set_target_difference = abs(average_time_to_set_target - target_average_time_to_set_target)
    
    
    
    return {'final_aware_total_percent': final_aware_total_percent,
            'final_committed_total_percent': final_committed_total_percent,
            'final_target_set_total_percent': final_target_set_total_percent,
        
            'aware_count_yearly': np.array(aware_count_yearly),
            'committed_count_yearly': np.array(committed_count_yearly),
            'target_set_count_yearly': np.array(target_set_count_yearly),
            
            'final_committed_difference': final_committed_difference,
            'final_target_set_difference': final_target_set_difference,
            'average_time_to_set_target_difference':average_time_to_set_target_difference,
            
            "average_time_to_set_target": average_time_to_set_target}



#%% CODE RUN 

num_companies= 50 # (real number 2233)
max_steps= 60


if __name__ == "__main__":

    ema_logging.log_to_stderr(ema_logging.INFO)
    
    py_model = Model('SBTiModel', function=sbti_model)
    
    py_model.constants = [
        Constant("leadership_lever", 1),
        Constant("risk_awareness_lever", 1),
        Constant("reputation_lever", 1),
        Constant("shareholder_pressure_lever", 1),
        Constant("manager_pressure_lever", 1),
        Constant("employee_pressure_lever", 1),
        Constant("market_pressure_lever", 1),
        Constant("num_companies", num_companies),
        Constant("max_steps", max_steps),
        Constant("pres_mot_eval", "product"),
        
        Constant("shareholder_pressure_coefficient", 0.253),
        Constant("manager_pressure_coefficient", 0.254),
        Constant("employee_pressure_coefficient", 0.238),
        Constant("market_pressure_coefficient", 0.210),
        Constant("manufacturing_coefficient", 0.232),
        Constant("non_manufacturing_coefficient", 0.358),
        Constant("max_comm_duration", 24),
        Constant("alpha", 0.15),
        Constant("beta", 0.15),
        Constant("gamma", 0.15),
        Constant("delta", 0.05),
        Constant("companies_turn_aware_per_round", 1),
        Constant('meetings_per_year',10),
        Constant('sigma',0.1)
        
        
    ]
    
    # most significant uncertainties according to the sensitivity analysis
    py_model.uncertainties = [
        # network
        IntegerParameter("rewiring_frequency", 1, 36),
        
        # awareness
        IntegerParameter("aware_update_step", 1, 28),
        IntegerParameter("steps_per_round", 1, 10),
        
        
        # commitment
        RealParameter("pressure_threshold", 1,  15),
        RealParameter("motivation_threshold", 1, 15),
    

        # set target
        CategoricalParameter("internal_target_range", [(0,0.5), (0,1), (0.5,1)]),
        RealParameter("work_rate", 0.1, 1),
        ]
      
    
    #platypus opt tries to minimize the difference the model's outcome of % committed, % set target 
    # and average time between commtiment and set target and real values
    py_model.outcomes = [

        ScalarOutcome("final_committed_difference", ScalarOutcome.MINIMIZE),        
        ScalarOutcome("final_target_set_difference", ScalarOutcome.MINIMIZE),

        ScalarOutcome("average_time_to_set_target_difference", ScalarOutcome.MINIMIZE),
        ]  
        # TimeSeriesOutcome("aware_count_yearly"),
        # TimeSeriesOutcome("committed_count_yearly"),
        # TimeSeriesOutcome("target_set_count_yearly"),
        



    def min_one_target_set(x):
        if np.isnan(x):
            return 1  # if x is NaN, the constraint is violated
        else:
            return 0  # if x is a number, the constraint is met
    
    constraints = [Constraint("At least one target set", 
                              outcome_names="average_time_to_set_target_difference", 
                              function=min_one_target_set)]

    
    
    #%% Calibration
    
    with MultiprocessingEvaluator(py_model) as evaluator:
        results = evaluator.optimize(nfe=200, searchover="uncertainties", epsilons=[0.01] * len(py_model.outcomes), constraints=constraints)
    
    print(results)
    
#%%    


    from ema_workbench import load_results
    
 #   results.to_csv('data_collected/results_calibration_0807_2333agents_100nfe_0.01epsilon.csv', index=False)
    #save_results(results, 'data_collected/results_calibration_0807_1000agents_1000nfe_0.1epsilon.tar.gz')
    
    
    cal_results = load_results('data_collected/results_calibration_0707.tar.gz')









