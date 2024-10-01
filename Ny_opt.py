# %% ----- Interchangeable data - User Interface ----- #

# Year to examine
year = 2050
# Use 2021, 2022 or 2023, earlier years could be implemented
# You can also use 2050 if you want to use a made-up scenario

# Zone within Norway to examine
zone = "NO1"
# Uke "NO1", "NO2", "NO3", "NO4", "NO5"
# if you chose 2050 (made rup scenario), it doesn't matter which zone you choose here

# Define the set of hours in the year (step1 = 1, tf = 8760)
# or define time interval to be examined
step1 = 0
tf = 8759

# Production price (cost) in NOK/MWh (marginal cost)
MC_base = 50.0  # Example
MC_TES = MC_base * 1.1  # Example
# TODO: find more correct numbers, var 100 før!


# Investment parameters
interest_rate = 0.05
investment_cost = 100 # NOK/MWh
lifetime = 40 # Years
# TODO: check if numbers are correct

# %% ----- Dependencies ----- #

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from Data_handeling import read_previous_spot

# %% ----- Model setup ----- #

# Create a concrete Pyomo model
model = pyo.ConcreteModel()

# List for hour range to be examined
hours = list(range(step1, tf + 1))

# %% ----- Reading spot price data from csv or excel----- #

power_prices = read_previous_spot(year, zone)

# %% ----- Set ----- #
model.time = pyo.Set(initialize=range(step1, tf + 1))

# %% ----- Variables ----- #

# TES size
model.mass_TES = pyo.Var(within = pyo.NonNegativeReals)

# Internal energy TES
model.u_TES = pyo.Var(model.time, within = pyo.NonNegativeReals)

# Mass flows
model.massflow_base =   pyo.Var(model.time, within = pyo.NonNegativeReals)
model.massflow_in =     pyo.Var(model.time, within = pyo.NonNegativeReals)
model.massflow_out =    pyo.Var(model.time, within = pyo.NonNegativeReals)
model.massflow_gen =    pyo.Var(model.time, within = pyo.NonNegativeReals)

'''
# Heat flows
model.heat_gen =        pyo.Var(model.time, within = pyo.NonNegativeReals)
model.heat_base =       pyo.Var(model.time, within = pyo.NonNegativeReals)
model.heat_in =         pyo.Var(model.time, within = pyo.NonNegativeReals)
model.heat_out =        pyo.Var(model.time, within = pyo.NonNegativeReals)
model.heat_turbine =    pyo.Var(model.time, within = pyo.NonNegativeReals)
'''

# Power
model.power_base = pyo.Var(model.time, within = pyo.NonNegativeReals)
model.power_TES = pyo.Var(model.time, within = pyo.NonNegativeReals)

# %% ----- Objective Function: Maximize the surplus (profit) ----- #

def objective_rule(model):
    return sum(
                - investment_cost * model.mass_TES * (interest_rate / (1 - (1 + interest_rate) ** (- lifetime)))
                + ( power_prices[hour] - MC_base ) * model.power_base[hour]
                + ( power_prices[hour] - MC_TES ) * model.power_TES[hour]  for hour in model.time)
model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)

# %% ----- Parameters ----- #

# TODO: find enthalpies, efficiencies, temperatures
# Max size on TES system
max_mass_TES = 10000000

# Enthalpies
h_turbine_input = 3100              # kJ/kg
h_turbine_output = 2200             # kJ/kg
h_TES_input = 3100                  # kJ/kg
h_TES_output = 2200                 # kJ/kg

# Initial internal energy
u_TES_init = 0                      # KJ

# Efficiencies
efficiency_turbine = 0.95
efficiency_charge = 0.97
efficiency_discharge = 0.93

# TES temperatures
temperature_max = 600 + 273.15      # K
temperature_min = 200 + 273.15      #K

# Specific heat capacity
specific_heat = 1.5                 # kJ / kg * K

# Max steam generation from reactor
reactor_gen_max = 1080000000        # kJ / h

# Ramping
pro = 5 # %/min
ramping = pro * 0.6 * reactor_gen_max  # [MW/h]
## (0.6 comes when converting from %/min to MW/h)

# %% ----- Constraints ----- #

# Base power generation
def base_power_rule(model, hour):
    return (model.power_base[hour] == model.massflow_base[hour] * (h_turbine_input - h_turbine_output))
model.base_power = pyo.Constraint(model.time, rule = base_power_rule)

# TES power generation
def TES_power_rule(model, hour):
    return (model.power_TES[hour] == model.massflow_out[hour] * (h_turbine_input - h_turbine_output))
model.TES_power = pyo.Constraint(model.time, rule = TES_power_rule)

# Energy balance in TES
def energy_balance_rule(model, hour):
    if hour == step1:
        return (model.u_TES[hour] ==    u_TES_init
                                        + model.massflow_in[hour] * h_TES_input * efficiency_charge
                                        - (model.massflow_out[hour] * h_TES_output)/efficiency_discharge)
    else:
        return (model.u_TES[hour] ==    model.u_TES[hour - 1]
                                        + model.massflow_in[hour] * h_TES_input * efficiency_charge
                                        - (model.massflow_out[hour] * h_TES_output)/efficiency_discharge)
model.energy_balance = pyo.Constraint(model.time, rule = energy_balance_rule)

# Max TES capacity
def max_TES_cap_rule(model, hour):
    return (model.u_TES[hour] <= model.mass_TES * specific_heat * (temperature_max - temperature_min))
model.max_TES_cap = pyo.Constraint(model.time, rule = max_TES_cap_rule)

# ramping power min
def min_ramping_rule(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return (-ramping <= model.massflow_gen[hour] - model.massflow_gen[hour - 1])
model.min_ramping = pyo.Constraint(model.time, rule = min_ramping_rule)

# ramping power max
def max_ramping_rule(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return ((model.massflow_gen[hour]) - (model.massflow_gen[hour - 1]) <= ramping)
model.max_ramping = pyo.Constraint(model.time, rule = max_ramping_rule)

# Max reactor generation
def max_generation_rule(model, hour):
    return (model.massflow_gen[hour] <= reactor_gen_max)
model.max_generation = pyo.Constraint(model.time, rule = max_generation_rule)

# Max mass for TES
def max_size_rule(model):
    return (model.mass_TES <= max_mass_TES)
model.max_size = pyo.Constraint(rule = max_size_rule)

# Min mass for TES
def min_size_rule(model):
    return (model.mass_TES >= 0) # TODO: set til noe annet enn 0
model.min_size = pyo.Constraint(rule = min_size_rule)

# mass conservation
def mass_conservation_rule(model, hour):
    return (model.massflow_gen[hour] == model.massflow_in[hour] + model.massflow_base[hour])
model.mass_conservaton = pyo.Constraint(model.time, rule = mass_conservation_rule)

# %% ----- Solving the optimization problem ----- #

opt = SolverFactory("gurobi", solver_io="python")
results = opt.solve(model, load_solutions = True)

print('\n')


print('\n')
results = opt.solve(model, load_solutions=True)
print(f'\nSolver Status: {results.solver.termination_condition}')



# %% ----- Printing and plotting results ----- #

print("Surplus: ", pyo.value(model.objective), "NOK")
print(f'Optmial TES size: {model.mass_TES.value} kg')

