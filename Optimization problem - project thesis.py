# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:58:33 2023

@author: agrot

Version 2.1.1 of the project thesis optimization problem code
- cleaning up and making some adjustments (pricing in obj.fnc., flow constraints)

This model code is far from finished.
Apologizes for the inconveniences caused by the lack of model build modularity

Most code for plots are included, but commented out because of the total 
additional run-time for plotting and the unclearity all the plots would create.
This allows for selection of plots based on needs.
"""

# %% ----- Interchangeable data - User Interface ----- #

# Year to examine
year = 2022 
# Use 2022 or 2023, earlier years could be implemented
# For 2022 hour [1, 8758], and for 2023 [1, 8352] are available

# Define the set of hours in the year (step1 = 1, tf = 8760)
# or define time interval to be examined
step1 = 1
tf = 8758 # not 8760 because of missing data
# TODO: Add the right data from the specialization project

# Defining reactor heat extraction limits
gen_maxcap = 300
gen_lowcap = 0
# TODO: check if numbers are correct

# Production price (cost) in NOK/MWh (marginal cost)
MC_gen = 100.0  # Example, will be lower
MC_turbo = MC_gen * 1.1  # Example
# TODO: find more correct numbers

# Defining ramping limits for electric generator
lower_ramp_lim = -5  # [%/min]
upper_ramp_lim = 5
# TODO: check if numbers are correct

# --- TES --- #

# TES storage duration
t_tes = 12  # [hours]
# TODO: check if numbers are correct

# Defining TES inflow limits
inflow_maxcap = 300  # [MW]
inflow_lowcap = 0
# TODO: check if numbers are correct

# Defining TES outflow limits
outflow_maxcap = 300  # [MW]
outflow_lowcap = 0
# TODO: check if numbers are correct

# Defining TES capacity
tes_maxcap = outflow_maxcap * t_tes  # [MWh]
tes_lowcap = 0
# TODO: check if numbers are correct


# %% ----- Dependencies ----- #

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


# %% ----- Model setup ----- #

# Create a concrete Pyomo model
model = pyo.ConcreteModel()

# List for hour range to be examined
# Assuming there are 8760 hours in a year (8761, one month: 745)
hours = range(step1, tf + 1)


# %% ----- Reading data ----- #

def InputData(data_file):  # reading the excel file with the values of generation and cost

    if year == 2022:
        col = 'B'
    elif year == 2023:
        col = 'C'
    inputdata = pd.read_excel(data_file, usecols=col, na_values='-').dropna()
    power_prices = inputdata['price ' + str(year)].tolist()

    return power_prices


power_prices = InputData('Power prices 8760 1.0.xlsx')

print(power_prices)


# %% ----- Variables ----- #

model.power_generation       = pyo.Var(hours, within=pyo.NonNegativeReals) # power generation in each hour
model.power_generation_state = pyo.Var(hours, within=pyo.Binary)           # power generation state in each hour
model.basepow                = pyo.Var(hours, within=pyo.NonNegativeReals) # base power extract in each hour
model.outflow_tes            = pyo.Var(hours, within=pyo.NonNegativeReals) # TES outflow in each hour
model.inflow_tes             = pyo.Var(hours, within=pyo.NonNegativeReals) # inflow to TES in each hour
model.inflow_tes_state       = pyo.Var(hours, within=pyo.Binary)           # inflow to TES state in each hour
model.fuel_tes               = pyo.Var(hours, within=pyo.NonNegativeReals) # fuel level in TES in each hour
model.tes_full_state         = pyo.Var(hours, within=pyo.Binary)           # tes full state

# %% ----- Objective Function: Maximize the surplus (profit) ----- #

def objective_rule(model):
    return sum((power_prices[hour] - MC_gen) * model.basepow[hour] 
               + model.outflow_tes[hour] * (power_prices[hour] - MC_turbo) for hour in hours)
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


# %% ----- Reactor Heat Extract Constraints ----- #

def production_limit_upper(model, hour):
    return model.basepow[hour] + model.inflow_tes[hour] <= gen_maxcap
model.prod_limup = pyo.Constraint(hours, rule=production_limit_upper)


def production_limit_lower(model, hour):
    return gen_lowcap <= model.basepow[hour] + model.inflow_tes[hour]
model.prod_limlow = pyo.Constraint(hours, rule=production_limit_lower)


# %% ----- Ramping Constraints ----- #


# Converting ramping limits
gen_lowramp = lower_ramp_lim * 0.6 * gen_maxcap  # [MW/h]
gen_maxramp = upper_ramp_lim * 0.6 * gen_maxcap
# TODO: Find out where 0.6 came from


def production_ramping_up(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return model.power_generation[hour] - model.power_generation[hour - 1] <= gen_maxramp
model.prod_rampup = pyo.Constraint(hours, rule=production_ramping_up)


def production_ramping_down(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return gen_lowramp <= model.power_generation[hour]- model.power_generation[hour - 1]

# %% ----- TES Energy Balance Constraint ----- #


def tes_balance(model, hour):
    if hour == step1:
        return model.fuel_tes[hour] == 0
    else:
        return model.fuel_tes[hour] == model.fuel_tes[hour - 1] + model.inflow_tes[hour - 1] - model.outflow_tes[hour - 1]
model.tes_energy_bal = pyo.Constraint(hours, rule=tes_balance)


# %% ----- TES Flow Constraints using big M ----- #

M = 10**4

# ----- Binary constraints ----- #

# Constraint to link binary variable with power generation


def binary_powgen(model, hour):
    return model.power_generation[hour] <= model.power_generation_state[hour] * M
model.binary_constraint1 = pyo.Constraint(hours, rule=binary_powgen)

# Constraint to link binary variable with TES inflow


def binary_inflow(model, hour):
    return model.inflow_tes[hour] <= model.inflow_tes_state[hour] * M
model.binary_constraint2 = pyo.Constraint(hours, rule=binary_inflow)


# %% ----- TES Capacity Limit state link and time constraint ----- #

# Constraint to link binary variable

def binary_tescap(model, hour):
    return model.fuel_tes[hour] >= tes_maxcap * model.tes_full_state[hour]
model.binary_constraintfullcap = pyo.Constraint(hours, rule=binary_tescap)

def binary_tescap2(model, hour):
    return model.fuel_tes[hour] <= (1 - model.tes_full_state[hour]) * (tes_maxcap - 1) + M * model.tes_full_state[hour]
model.binary_constraintfullcap2 = pyo.Constraint(hours, rule=binary_tescap2)


# %% ----- TES Flow Limit Constraints ----- #

def tes_inflow_limit_upper(model, hour):
    return model.inflow_tes[hour] <= inflow_maxcap
model.tesin_limup = pyo.Constraint(hours, rule=tes_inflow_limit_upper)


def tes_inflow_limit_lower(model, hour):
    return inflow_lowcap <= model.inflow_tes[hour]
model.tesin_limlow = pyo.Constraint(hours, rule=tes_inflow_limit_lower)


# ----- TES production limit constraints ----- #

def tes_outflow_limit_upper(model, hour):
    return model.outflow_tes[hour] <= outflow_maxcap
model.tesout_limup = pyo.Constraint(hours, rule=tes_outflow_limit_upper)


def tes_outflow_limit_lower(model, hour):
    return outflow_lowcap <= model.outflow_tes[hour]
model.tesout_limlow = pyo.Constraint(hours, rule=tes_outflow_limit_lower)


# %% ----- TES Capacity Constraint ----- #

def tes_cap_upper(model, hour):
    return model.fuel_tes[hour] <= tes_maxcap
model.tes_up_cap = pyo.Constraint(hours, rule=tes_cap_upper)


def tes_cap_lower(model, hour):
    return tes_lowcap <= model.fuel_tes[hour]
model.tes_low_cap = pyo.Constraint(hours, rule=tes_cap_lower)


# %% ----- Combining power constraint ----- #

def comb_power(model, hour):
    return model.power_generation[hour] == model.basepow[hour] + model.outflow_tes[hour]
model.combining_power = pyo.Constraint(hours, rule=comb_power)








# %% ----- Solving the optimization problem ----- #

opt = SolverFactory("gurobi", solver_io="python")
#opt.options['tee'] = True
results = opt.solve(model, load_solutions = True)

print('\n')

# %% ----- Printing and plotting results ----- #

print("Optimal Surplus: ", pyo.value(model.objective), "NOK")

# %% ----- Calculating value factor ----- #

sumprod = 0 
nprod = 0 

for h in hourslist:
    if model.power_generation[h].value > 0:
        sumprod += power_prices[h]
        nprod += 1
        
C_avgprod = sumprod/nprod
C_avg = sum(power_prices)/len(power_prices)

vf = C_avgprod/C_avg
print(vf)

# %% ----- Calculating capacity factor ----- #

totsum = 0

for h in hourslist:
    totsum += model.power_generation[h].value
    
cf = totsum/((tf+1-step1)*gen_maxcap)
print(cf)
