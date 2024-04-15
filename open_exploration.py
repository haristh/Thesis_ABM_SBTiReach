#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:03:06 2023

@author: charistheodorou

This file includes the open exploration parts of the model. EMA Workbench library 
is imported to use Patient Rule Induction Method (PRIM) and Feature Scoring. This is
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
from scipy import stats
import seaborn as sns
sns.set_style('white')
import seaborn as sns
import pandas as pd
import numpy as np




# Set the display options- needed to show all values in the console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


#%% Function prepared to be used for EMA workbench tools


def sbti_model(num_companies= 1000, max_steps= 60,
             
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

    
    
    # Create a new instance of the model with the given parameters, imported from SBTiModel_main
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
    
    #placeholders for the yearly number of aware, committed and set target companies (not used in the end)
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
    
    
    # The following call functions from the SBTiModel to collect data at the end of the run
    #total number of companies at each state at the end of the run
    final_aware_total = model.get_aware_total()
    final_committed_total = model.get_committed_total()
    final_target_set_total = model.get_target_set_total()

    #percentage of companies at each state at the end of the run
    final_aware_total_percent = model.get_aware_total_percent()
    final_committed_total_percent = model.get_committed_total_percent()
    final_target_set_total_percent = model.get_target_set_total_percent()

    # Functions from the SBTiModel counting number of aware, 
    # committed and with target companies per sector and per country
    final_aware_by_country = model.get_aware_by_country()
    final_aware_by_sector = model.get_aware_by_sector()
    
    final_committed_by_country = model.get_committed_by_country()
    final_committed_by_sector = model.get_committed_by_sector()
    
    final_target_set_by_country = model.get_target_set_by_country()
    final_target_set_by_sector = model.get_target_set_by_sector()
    
    
    #function from the SBTi the calculates the average time from commitment to set target
    average_time_to_set_target = model.average_time_to_set_target()
    
    # Calculate absolute differences from target values (not used in the end, implemented in platypus)
    committed_diff = int(abs(final_committed_total - 0.28 * num_companies))
    target_set_diff = int(abs(final_target_set_total - 0.14 * num_companies))
    
    
    return {'final_aware_total': final_aware_total,
            'final_committed_total': final_committed_total,
            'final_target_set_total': final_target_set_total,
            
            'final_aware_total_percent': final_aware_total_percent,
            'final_committed_total_percent': final_committed_total_percent,
            'final_target_set_total_percent': final_target_set_total_percent,
        
            'aware_count_yearly': np.array(aware_count_yearly),
            'committed_count_yearly': np.array(committed_count_yearly),
            'target_set_count_yearly': np.array(target_set_count_yearly),
            
            'final_aware_by_country': final_aware_by_country,
            'final_aware_by_sector': final_aware_by_sector,
            
            'final_committed_by_country': final_committed_by_country,
            'final_committed_by_sector': final_committed_by_sector,
            
            'final_target_set_by_country': final_target_set_by_country,
            'final_target_set_by_sector': final_target_set_by_sector,
            'committed_diff': committed_diff,
            'target_set_diff': target_set_diff,
            
            
            "average_time_to_set_target": average_time_to_set_target}
    


if __name__ == "__main__":
        
    n_exp = 1000 # number of experiments
    num_companies= 1000 
    max_steps= 600 # 2015-2020, explained in the report
    
    
    ema_logging.log_to_stderr(ema_logging.INFO)
   
    py_model = Model('SBTiModel', function=sbti_model) # instantiate the model
    
    
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
      
    ]
    
    py_model.uncertainties = [
                    # network
                    RealParameter("alpha", 0.01, 0.20),
                    RealParameter("beta", 0.01, 0.20),
                    RealParameter("gamma", 0.01, 0.20),
                    RealParameter("delta", 0.005, 0.10),
                    IntegerParameter("rewiring_frequency", 1, 36),
                    
                    # awareness
                    IntegerParameter("aware_update_step", 1, 30),
                    IntegerParameter("companies_turn_aware_per_round", 1, 3),
                    IntegerParameter("steps_per_round", 1, 30),
                    
                    
                    # commitment
                    RealParameter("shareholder_pressure_coefficient", 0.1, 2.0),
                    RealParameter("manager_pressure_coefficient", 0.1, 2.0),
                    RealParameter("employee_pressure_coefficient", 0.1, 2.0),
                    RealParameter("market_pressure_coefficient", 0.1, 2.0),
                    RealParameter("manufacturing_coefficient", 0.1, 2.0),
                    RealParameter("non_manufacturing_coefficient", 0.1, 2.0),
                    
                    RealParameter("pressure_threshold", 1, 20),
                    RealParameter("motivation_threshold", 1, 20),
                    IntegerParameter("max_comm_duration", 12, 48),
                
                    IntegerParameter("meetings_per_year", 1, 20),
                    CategoricalParameter("pres_mot_eval", ["product", "sum", "serial"]),
                    
                    # setting target
                    CategoricalParameter("internal_target_range", [(0,0.5), (0,1), (0.5,1)]),
                    RealParameter("work_rate", 0.1, 1),

                    
                    RealParameter("sigma", 0.01, 0.1)
                    ]
    

    py_model.outcomes = [
            ScalarOutcome("final_aware_total"),
            ScalarOutcome("final_committed_total"),
            ScalarOutcome("final_target_set_total"),
            
            ScalarOutcome("final_aware_total_percent"),
            ScalarOutcome("final_committed_total_percent"),
            ScalarOutcome("final_target_set_total_percent"),
            
            TimeSeriesOutcome("aware_count_yearly"),
            TimeSeriesOutcome("committed_count_yearly"),
            TimeSeriesOutcome("target_set_count_yearly"),
            
            ArrayOutcome("final_aware_by_country"),
            ArrayOutcome("final_aware_by_sector"),
            ArrayOutcome("final_committed_by_country"),
            ArrayOutcome("final_committed_by_sector"),
            ArrayOutcome("final_target_set_by_country"),
            ArrayOutcome("final_target_set_by_sector"),
            
            ScalarOutcome("committed_diff"),
            ScalarOutcome("target_set_diff"),
            ScalarOutcome("average_time_to_set_target")
            ]                    
                                         

    
#%% Run model for n number of experiments. Model was run with 1000 companies due to computational limitations    

    with MultiprocessingEvaluator(py_model) as evaluator: #, n_processes=-1
        results = evaluator.perform_experiments(scenarios = n_exp) # policies = 2, uncertainty_sampling = LHS)
    

    
    
    #%% saving results - commented out to avoid overwriting
    
    #save_results(results, 'data_collected/results_open_exploration_0307_2_'+str(num_companies)+'agents_'+ str(n_exp)+'experiments.tar.gz')
        
    #%% Load results
    from ema_workbench import load_results
        
    results = load_results('data_collected/results_open_exploration_0307_2_1000agents_1000experiments.tar.gz') 
    experiments, outcomes = results

    #%% 1. PRIM analysis- Average time to set target
    # Increase size of labels, ticks, and title
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    
    # The estimated time from commitment to setting a target is 6.4 months
    # outcome of interest then is decided to be between 4 and 6 months
    
    outcome_of_interest =  (outcomes['average_time_to_set_target'] > 5) & (outcomes['average_time_to_set_target'] <7)
    x = outcomes['average_time_to_set_target']
    
    # Perform PRIM analysis
    prim_alg = prim.Prim(experiments, outcome_of_interest, threshold=0.6, peel_alpha=0.1)
    box1 = prim_alg.find_box()
    
    
    box1.show_tradeoff()
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMtradeoff_committed_5to7months_Report.png', dpi = 300)
    plt.show()
    
    # Inspect the results
    box1.inspect_tradeoff()
    plt.show()
        
        
    box1.inspect()
    box1.inspect( style="graph")
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMvariables_committed_5to7months_Report.png', dpi = 300)
    plt.show()
    
    #box1.show_pairs_scatter()
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMpairsscatterplot_committed_5to7months_Report.png', dpi = 500)
    #plt.show()
    

    
    
    #%% 2. PRIM analysis- Percentage of companies that committed
    
    # The estimated percentage of companies by 2021 is 20% 
    # outcome of interest then is decided to be between 15% and 30%
    
    outcome_of_interest = (outcomes['final_committed_total_percent'] > 0.10) & (outcomes['final_committed_total_percent'] < 0.30) 
    y = outcomes['final_committed_total_percent']
    
    
    # Perform PRIM analysis
    prim_alg = prim.Prim(experiments, outcome_of_interest, threshold=0.4, peel_alpha=0.1)
    
    box1 = prim_alg.find_box()
    
    
    box1.show_tradeoff()
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMtradeoff_committed_10to30percent_Report.png', dpi = 300)
    plt.show()
    
    # Inspect the results
    #box1.inspect_tradeoff()
    #plt.show()
        
        
    print(box1.inspect())
    box1.inspect( style="graph")
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMvariables_committed_10to30percent_Report.png', dpi = 300)
    plt.show()
    
    # box1.show_pairs_scatter()
    # #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMpairsscatterplot_committed_below40percent.png', dpi = 300)
    # plt.show()
    
    #%% 3. PRIM analysis- Percentage of companies that set targets
    
    # The estimated percentage of companies by 2021 is 12% 
    # outcome of interest then is decided to be between 1% and 30% months
    
    outcome_of_interest = (outcomes['final_target_set_total_percent'] > 0.01) & (outcomes['final_target_set_total_percent'] < 0.2) 
    z = outcomes['final_target_set_total_percent']
    
    # Perform PRIM analysis
    prim_alg = prim.Prim(experiments, outcome_of_interest, threshold=0.4, peel_alpha=0.1)
    box1 = prim_alg.find_box()
    
    
    box1.show_tradeoff()
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMtradeoff_set_target_0to20percent_Report.png', dpi = 300)
    plt.show()
    
    # Inspect the results
    box1.inspect_tradeoff()
    plt.show()
        
        
    box1.inspect()
    box1.inspect( style="graph")
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMvariables_set_target_0to20percent_Report.png', dpi = 300)
    #plt.savefig('sreport_images/ensitivity_analysis_open_exploration/PRIM_committed_below40percent.png', dpi = 300)
    plt.show()
    
    #box1.show_pairs_scatter()
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMpairsscatterplot_committed_below40percent.png', dpi = 300)
    #plt.show()



    #%% 4. PRIM- All three
    
    
    outcome_of_interest = ((outcomes['average_time_to_set_target'] > 5) & (outcomes['average_time_to_set_target'] <7)
                           & (outcomes['final_committed_total_percent'] > 0.10) & (outcomes['final_committed_total_percent'] < 0.30) 
                           & (outcomes['final_target_set_total_percent'] > 0.01) & (outcomes['final_target_set_total_percent'] < 0.2)) 
    y = outcomes['final_target_set_total_percent']
    
    # Perform PRIM analysis
    prim_alg = prim.Prim(experiments, outcome_of_interest, threshold=0.3, peel_alpha=0.1)
    box1 = prim_alg.find_box()
    
    
    box1.show_tradeoff()
    plt.show()
    
    # Inspect the results
    box1.inspect_tradeoff()
    plt.show()
        
        
    #box1.inspect()
    box1.inspect( style="graph")
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIM_committed_below40percent.png', dpi = 300)
    plt.show()
    
    box1.show_pairs_scatter()
    #plt.savefig('report_images/sensitivity_analysis_open_exploration/PRIMpairsscatterplot_committed_below40percent.png', dpi = 300)
    plt.show()




#%% Feature Scoring for all three outcomes discussed above
  

    from ema_workbench.analysis import feature_scoring

    x = experiments
    
    # scalar outcomes
    y_scalars = {key: outcomes[key] for key in ['final_aware_total_percent', 'final_committed_total_percent', 
                                                 'final_target_set_total_percent']}
    
    
    fs_scalars = feature_scoring.get_feature_scores_all(x, y_scalars)
    print(fs_scalars)
    plt.figure(figsize=(10, 15))
    sns.heatmap(fs_scalars, cmap="viridis", annot=True)
    plt.xticks(rotation= 45)
    
    plt.tight_layout()
   # plt.savefig('report_images/sensitivity_analysis_open_exploration/feature_scoring_Report.png', dpi = 500)
    plt.show()

