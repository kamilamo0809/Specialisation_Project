# %% ----- Interchangeable data - User Interface ----- #

def massflow_to_power(massflow):
    return massflow * 60 * 60 * (h_turbine_input - h_turbine_output)

def power_to_massflow(power):
    return power / (60 * 60 * (h_turbine_input - h_turbine_output))

def joules_to_watt(joules):
    return joules / (60 * 60)

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

# Marginal cost of productionin NOK/W (marginal cost)
MC_base = 0.0001  # Example
MC_TES = MC_base * 1.1  # Example
# TODO: find more correct numbers, var 100 før!


# Investment parameters
interest_rate = 0.05
investment_cost = 20 # NOK/kg
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

# Power
model.power_base = pyo.Var(model.time, within = pyo.NonNegativeReals)
model.power_TES = pyo.Var(model.time, within = pyo.NonNegativeReals)

# %% ----- Objective Function: Maximize the surplus (profit) ----- #

def objective_rule(model):
    return sum(- investment_cost * model.mass_TES * (interest_rate / (1 - (1 + interest_rate) ** (- lifetime)))
                + ( power_prices[hour] - MC_base ) * model.power_base[hour]
                + ( power_prices[hour] - MC_TES ) * model.power_TES[hour]  for hour in model.time)
model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)

# %% ----- Parameters ----- #

# TODO: find enthalpies, efficiencies, temperatures
# Max size on TES system
max_mass_TES = 10000000

# Enthalpies
h_turbine_input = 3100              # J/kg
h_turbine_output = 2200             # J/kg
h_TES_input = 3100                  # J/kg
h_TES_output = 2200                 # J/kg

# Initial internal energy
u_TES_init = 0                      # J

# Efficiencies
efficiency_turbine = 0.9
efficiency_charge = 0.85
efficiency_discharge = 0.85

# TES temperatures
temperature_max = 600 + 273.15      # K
temperature_min = 200 + 273.15      # K

# Specific heat capacity
specific_heat = 1500                 # J / kg * K

# Max steam generation from reactor
massflow_gen_max = power_to_massflow(300000000)  # kg/h

# Ramping
pro = 0.02 # %/min
ramping = pro * 0.6 * massflow_gen_max  # kg/h

discharging_power = 400000000 # MW


# %% ----- Constraints ----- #

# Base power generation
def base_power_rule(model, hour):
    return model.power_base[hour] == massflow_to_power(model.massflow_base[hour])
model.base_power = pyo.Constraint(model.time, rule = base_power_rule)

# TES power generation
def TES_power_rule(model, hour):
    return model.power_TES[hour] == massflow_to_power(model.massflow_out[hour])
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

def internal_energy_end_rule(model):
    return model.u_TES[tf] == u_TES_init
model.internal_energy_end = pyo.Constraint(rule = internal_energy_end_rule)

# Max TES capacity
def max_TES_cap_rule(model, hour):
    return (model.u_TES[hour] <= model.mass_TES * specific_heat * (temperature_max - temperature_min))
model.max_TES_cap = pyo.Constraint(model.time, rule = max_TES_cap_rule)

# ramping power min
def min_ramping_rule(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return -ramping <= massflow_to_power(model.massflow_gen[hour] - model.massflow_gen[hour - 1])
model.min_ramping = pyo.Constraint(model.time, rule = min_ramping_rule)

# ramping power max
def max_ramping_rule(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return massflow_to_power(model.massflow_gen[hour] - model.massflow_gen[hour - 1]) <= ramping
model.max_ramping = pyo.Constraint(model.time, rule = max_ramping_rule)

# Max reactor generation
def max_generation_rule(model, hour):
    return model.massflow_gen[hour] <= massflow_gen_max
model.max_generation = pyo.Constraint(model.time, rule = max_generation_rule)

# Max mass for TES
def max_size_rule(model):
    return (model.mass_TES <= max_mass_TES)
model.max_size = pyo.Constraint(rule = max_size_rule)

# mass conservation
def mass_conservation_rule(model, hour):
    return (model.massflow_gen[hour] == model.massflow_in[hour] + model.massflow_base[hour])
model.mass_conservaton = pyo.Constraint(model.time, rule = mass_conservation_rule)

'''
# Charging constraint:
def charging_rule(model, hour):
    return (model.massflow_in[hour] <= model.mass_TES * specific_heat * (temperature_max - temperature_min))
model.charging = pyo.Constraint(model.time, rule = charging_rule)
'''
# TODO: make this constraint better

# TODO: make a discharge constraint!

'''
# Charging constraint:
def discharging_rule(model, hour):
    return (model.power_TES[hour] <= 400000000)
model.discharging = pyo.Constraint(model.time, rule = discharging_rule)
'''

# Sum
def power_sum_rule(model):
    return sum(model.massflow_in[hour] * h_TES_input * efficiency_charge
            - (model.massflow_out[hour] * h_TES_output)/efficiency_discharge for hour in model.time) == 0

# Mass flow constraints:

def discharging_power_rule(model, hour):
    return massflow_to_power(model.massflow_out[hour]) <= discharging_power
model.discharging_power = pyo.Constraint(model.time, rule = discharging_power_rule)

# %% ----- Solving the optimization problem ----- #

opt = SolverFactory("gurobi", solver_io="python")
results = opt.solve(model, load_solutions = True)

print('\n')


print('\n')
results = opt.solve(model, load_solutions=True)
print(f'\nSolver Status: {results.solver.termination_condition}')



# %% ----- Printing results ----- #

storage_cap = pyo.value(model.mass_TES) * specific_heat * (temperature_max - temperature_min)

surplus_battery = sum((pyo.value(power_prices[hour]) - MC_TES ) * pyo.value(model.power_TES[hour])  for hour in hours)
surplus_base = sum((pyo.value(power_prices[hour]) - MC_base ) * pyo.value(model.power_base[hour])  for hour in hours)

print("Surplus from battery storage: ", round(surplus_battery / 1e6, 2), "MNOK")
print("Surplus from base production: ", round(surplus_base / 1e6, 2), "MNOK")
print("Total surplus: ", round(pyo.value(model.objective) / 1e6, 2), "MNOK")
print(f'Optmial TES size: {round(model.mass_TES.value / 1e3, 3)} tons')
print(f'TES storage capacity: {round(storage_cap / 1000000, 2)} MW')

# %% ----- Plotting results ----- #

hourslist = list(hours)
massflow_base = list(pyo.value(model.massflow_base[hour]) for hour in hours)
massflow_out = list(pyo.value(model.massflow_out[hour]) for hour in hours)
massflow_in = list(pyo.value(model.massflow_in[hour]) for hour in hours)
u = list(pyo.value(model.u_TES[hour]) for hour in hours)
power_TES = list(pyo.value(model.power_TES[hour] / 1000000) for hour in hours)
power_base = list(pyo.value(model.power_base[hour] / 1000000) for hour in hours)
battery_energy = list((pyo.value(model.u_TES[hour]) / 1000000) for hour in hours)

def make_monthly_list(yearly_list):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    all_hours_in_year = []
    start_idx = 0
    for month in days_in_month:
        days = []
        all_hours_in_year.append(yearly_list[start_idx:start_idx + month])
        start_idx += 1
    return all_hours_in_year

def plot_power_output(list_of_mont_numbers):
    time = make_monthly_list(hours)
    TES = make_monthly_list(power_TES)
    base = make_monthly_list(power_base)

    for i in list_of_mont_numbers:
        plt.plot(time[i - 1], TES[i - 1], color = "hotpink", label = "power from battery storage")
        plt.plot(time[i - 1], base[i - 1], color = "mediumpurple", label = "base power")
        plt.xlabel("Time [h]")
        plt.ylabel("Power [MW]")
        plt.title(f"Power output for month number {i}")
        plt.show()

def plot_battery_storage():
    plt.plot(hourslist, battery_energy, color = 'dodgerblue')
    plt.fill_between(hourslist, battery_energy, color = 'skyblue', alpha = 0.4)
    plt.title("Stored energy in TES through the year")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [MW]")
    plt.legend()
    plt.show()

#plot_battery_storage()

# TODO: husk at batteriet har en "duration" siden varme ikke kan lagres særlig lenge