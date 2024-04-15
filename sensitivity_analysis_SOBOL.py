#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:06:23 2023

@author: charistheodorou

This file includes the global sensitivity analysis parts of the model. EMA Workbench library 
is imported to use SOBOL indices. This is
discussed in Chapter 4: Verification, Sensitivity analysis and Parameter Setup.


Running the data collection, saving figures and saving any data is commented out to avoid overwriting


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

from ema_workbench.analysis import pairs_plotting, prim

from SALib.analyze import sobol
from ema_workbench import Samplers
from ema_workbench.em_framework.salib_samplers import get_SALib_problem


import matplotlib.pyplot as plt
import statsmodels.api as sm
from SBTiModel_main import SBTiModel
from open_exploration import sbti_model

from scipy import stats
import seaborn as sns
sns.set_style('white')
import seaborn as sns
import pandas as pd
import numpy as np

#%% 
n_exp = 500 
num_companies= 500 
max_steps= 60


if __name__ == "__main__":

    ema_logging.log_to_stderr(ema_logging.INFO)
    
    py_model = Model('SBTiModel', function=sbti_model)
    
    # parameters that are not varied
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
    
    
    # parameters allowed to vary between the specified values
    py_model.uncertainties = [
        # network
        IntegerParameter("rewiring_frequency", 1, 36),
        
        # awareness
        IntegerParameter("aware_update_step", 10, 28),
        IntegerParameter("steps_per_round", 1, 10),
        
        
        # commitment
        RealParameter("pressure_threshold", 1,  15),
        RealParameter("motivation_threshold", 1, 15),
    

        # set target
        CategoricalParameter("internal_target_range", [(0,0.5), (0,1), (0.5,1)]),
        RealParameter("work_rate", 0.1, 1),
        ]
      
    
    #the outcomes that need to be collected by the ema_workbench
    py_model.outcomes = [
        
        ScalarOutcome("final_aware_total_percent"),
        ScalarOutcome("final_committed_total_percent"),
        ScalarOutcome("final_target_set_total_percent"),
        
        
        ScalarOutcome("average_time_to_set_target"),
        ]  


                  
    
    #%% SOBOL
    # running the evaluator. it was run for 100 and 500 agents for 500 expeirments
    # with MultiprocessingEvaluator(py_model) as evaluator:
    #       sa_results = evaluator.perform_experiments(scenarios=n_exp, uncertainty_sampling=Samplers.SOBOL)
    
    
#%%    


    from ema_workbench import load_results
    
    #save_results(sa_results, 'data_collected/results_sobol_0307_2_500_agents_500_experiments_largerthresholdsforaware_onlyproduct.tar.gz')
    
    
    sa_results = load_results('data_collected/results_sobol_0307_2_500_agents_500_experiments_largerthresholdsforaware_onlyproduct.tar.gz')
 
#%% Awareness percentage
    experiments_SOBOL, outcomes_SOBOL = sa_results

    problem = get_SALib_problem(py_model.uncertainties)
    Si = sobol.analyze(problem, outcomes_SOBOL["final_aware_total_percent"], calc_second_order=True, print_to_console=False)
    
    scores_filtered = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
    Si_df = pd.DataFrame(scores_filtered, index=problem["names"])
    
    sns.set_style("white")
    fig, ax = plt.subplots(1)
    
    indices = Si_df[["S1", "ST"]]
    err = Si_df[["S1_conf", "ST_conf"]]
    
    indices.plot.bar(yerr=err.values.T, ax=ax)
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(bottom=0.3)
    plt.show()
    
    # Sort by the ST score and select the top 10
    Si_df_sorted = Si_df.sort_values(by='ST', ascending=False)
    Si_df_top10 = Si_df_sorted.head(10)

    # Plot
    fig, ax = plt.subplots(1)
    indices = Si_df_top10[["S1", "ST"]]
    err = Si_df_top10[["S1_conf", "ST_conf"]]
    
    indices.plot.bar(yerr=err.values.T, ax=ax)
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(bottom=0.3)
    plt.xticks(rotation= 45)
    plt.show()
    
    
    print(Si_df)
 
    
#%% Committed percentage
    experiments_SOBOL, outcomes_SOBOL = sa_results
    
    
    
    problem = get_SALib_problem(py_model.uncertainties)
    Si = sobol.analyze(problem, outcomes_SOBOL["final_committed_total_percent"], calc_second_order=True, print_to_console=False)
    
    scores_filtered = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
    Si_df = pd.DataFrame(scores_filtered, index=problem["names"])
    
    sns.set_style("white")
    fig, ax = plt.subplots(1)
    
    indices = Si_df[["S1", "ST"]]
    err = Si_df[["S1_conf", "ST_conf"]]
    
    indices.plot.bar(yerr=err.values.T, ax=ax)
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(bottom=0.3)
    plt.show()
    
    # Sort by the ST score and select the top 10
    Si_df_sorted = Si_df.sort_values(by='ST', ascending=False)
    Si_df_top10 = Si_df_sorted.head(10)

    # Plot
    fig, ax = plt.subplots(1)
    indices = Si_df_top10[["S1", "ST"]]
    err = Si_df_top10[["S1_conf", "ST_conf"]]
    
    indices.plot.bar(yerr=err.values.T, ax=ax)
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(bottom=0.3)
    plt.xticks(rotation= 45)
    plt.show()
    
    
    print(Si_df)

#%%    Set target Percentage

    
    problem = get_SALib_problem(py_model.uncertainties)
    Si_target = sobol.analyze(problem, outcomes_SOBOL["final_target_set_total_percent"], calc_second_order=True, print_to_console=False)
    
    scores_filtered_target = {k: Si_target[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
    Si_df_target = pd.DataFrame(scores_filtered_target, index=problem["names"])
    
    sns.set_style("white")
    fig, ax = plt.subplots(1)
    
    indices = Si_df_target[["S1", "ST"]]
    err = Si_df_target[["S1_conf", "ST_conf"]]
    
    indices.plot.bar(yerr=err.values.T, ax=ax)
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(bottom=0.3)
    plt.show()
    
    # Sort by the ST score and select the top 10
    Si_df_sorted_target = Si_df_target.sort_values(by='ST', ascending=False)
    Si_df_top10_target = Si_df_sorted_target.head(10)

    # Plot
    fig, ax = plt.subplots(1)
    indices = Si_df_top10_target[["S1", "ST"]]
    err = Si_df_top10_target[["S1_conf", "ST_conf"]]
    
    indices.plot.bar(yerr=err.values.T, ax=ax)
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(bottom=0.3)
    plt.xticks(rotation= 45)
    plt.show()
    
    
    print(Si_df_target)
