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

# Marginal cost of productionin NOK/W (marginal cost)
MC_base = 0.0001  # Example
MC_TES = MC_base * 1.1  # Example
# TODO: find more correct numbers, var 100 før!


# Investment parameters
interest_rate = 0.05
investment_cost = 20 # NOK/kg # TODO: sett til 20
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

# Charging
model.is_charging = pyo.Var(model.time, domain = pyo.Binary)

# Internal energy TES
model.u_TES = pyo.Var(model.time, within = pyo.NonNegativeReals)
model.u_TES_max = pyo.Var(within = pyo.NonNegativeReals)

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
                + ( power_prices[hour] - MC_TES ) * model.power_TES[hour] for hour in model.time)
model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)

# %% ----- Parameters ----- #

# TODO: find enthalpies, efficiencies, temperatures
# Max size on TES system
max_mass_TES = 10000 #10000000

# Enthalpies
h_turbine_in = 3100           # kJ/kg
h_turbine_out = 2200          # kJ/kg

h_delta_turbine = h_turbine_in - h_turbine_out

h_charge_in = 2760            # kJ/kg
h_charge_out = 650            # kJ/kg

h_discharge_in = 2774         # kJ/kg
h_discharge_out = 1000        # kJ/kg

h_delta_charge = h_charge_in - h_charge_out
h_delta_discharge = h_discharge_in - h_discharge_out

# Initial internal energy
u_TES_init = 0                      # kWh

# Efficiencies
efficiency_turbine = 0.9
efficiency_charge = 0.9
efficiency_discharge = 0.9

# TES temperatures
temperature_max = 700 + 273.15      # K
temperature_min = 200 + 273.15      # K

# Specific heat capacity
specific_heat = 1                 # kJ / kg * K

# Max steam generation from reactor
massflow_gen_max = 300000 / h_delta_turbine  # kg/s

# Ramping
pro = 1 # %/min
ramping = pro * 0.6 * massflow_gen_max  # kg/h

charge_cap = 200000 / h_delta_turbine # kg/s # TODO finn ut hva charge og discharge power er
discharge_cap = 200000 / h_delta_turbine # kg/s


# %% ----- Constraints ----- #

# Base power generation
def base_power_rule(model, hour):
    return model.power_base[hour] == model.massflow_base[hour] * h_delta_turbine
model.base_power = pyo.Constraint(model.time, rule = base_power_rule)

# TES power generation
def TES_power_rule(model, hour):
    return model.power_TES[hour] == model.massflow_out[hour] * h_delta_turbine
model.TES_power = pyo.Constraint(model.time, rule = TES_power_rule)

# Energy balance in TES
def energy_balance_rule(model, hour):
    if hour == step1:
        return model.u_TES[hour] == u_TES_init + model.massflow_in[hour] * h_delta_charge * efficiency_charge

    else:
        return (model.u_TES[hour] ==    model.u_TES[hour - 1]
                                        + model.massflow_in[hour] * h_delta_charge * efficiency_charge
                                        - (model.massflow_out[hour] * h_delta_discharge)/efficiency_discharge)
model.energy_balance = pyo.Constraint(model.time, rule = energy_balance_rule)

# End internal energy
def internal_energy_end_rule(model):
    return model.u_TES[tf] == u_TES_init
model.internal_energy_end = pyo.Constraint(rule = internal_energy_end_rule)

# Connot discharge the first hour
def discharge_first_hour(model):
    return model.massflow_out[step1] == 0
model.discharge_first_hour = pyo.Constraint(rule = discharge_first_hour)

# Connot charge and discharge at the same time
def charge_power_cap_rule(model, hour):
    return model.massflow_in[hour] <= charge_cap * model.is_charging[hour]
model.charge_power_cap_rule = pyo.Constraint(model.time, rule = charge_power_cap_rule)

def discharge_power_cap_rule(model, hour):
    return model.massflow_out[hour] <= discharge_cap * (1 - model.is_charging[hour])
model.discharge_power_cap_rule = pyo.Constraint(model.time, rule = discharge_power_cap_rule)

# Max TES capacity
def max_TES_cap_rule(model, hour):
    return (model.u_TES[hour] <= model.u_TES_max)
model.max_TES_cap = pyo.Constraint(model.time, rule = max_TES_cap_rule)

def max_TES(model):
    return (model.u_TES_max == model.mass_TES * specific_heat * (temperature_max - temperature_min)) # TODO: finn ut om det skal vare * 3600 her
model.max_TES = pyo.Constraint(rule = max_TES)

# ramping power min
def min_ramping_rule(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return -ramping <= model.massflow_gen[hour] - model.massflow_gen[hour - 1]
model.min_ramping = pyo.Constraint(model.time, rule = min_ramping_rule)

# ramping power max
def max_ramping_rule(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return model.massflow_gen[hour] - model.massflow_gen[hour - 1] <= ramping
model.max_ramping = pyo.Constraint(model.time, rule = max_ramping_rule)

# Max reactor generation
def max_generation_rule(model, hour):
    return model.massflow_gen[hour] <= massflow_gen_max
model.max_generation = pyo.Constraint(model.time, rule = max_generation_rule)

# Max mass for TES
def max_size_rule(model):
    return model.mass_TES <= max_mass_TES
model.max_size = pyo.Constraint(rule = max_size_rule)

# mass conservation
def mass_conservation_rule(model, hour):
    return (model.massflow_gen[hour] == model.massflow_in[hour] + model.massflow_base[hour])
model.mass_conservaton = pyo.Constraint(model.time, rule = mass_conservation_rule)

# TODO: make this constraint better

# Disharging constraint:
def discharging_rule(model, hour):
    return (model.massflow_out[hour] * h_delta_discharge)/efficiency_discharge <= model.u_TES_max
model.discharging = pyo.Constraint(model.time, rule = discharging_rule)
# TODO: Make a better function for discharging

# %% ----- Solving the optimization problem ----- #

opt = SolverFactory("gurobi", solver_io="python")
results = opt.solve(model, load_solutions = True)

print('\n')


print('\n')
results = opt.solve(model, load_solutions=True)
print(f'\nSolver Status: {results.solver.termination_condition}')



# %% ----- Printing results ----- #

surplus_battery = sum((pyo.value(power_prices[hour]) - MC_TES ) * pyo.value(model.power_TES[hour])  for hour in hours)
surplus_base = sum((pyo.value(power_prices[hour]) - MC_base ) * pyo.value(model.power_base[hour])  for hour in hours)

print("Surplus from battery storage: ", round(surplus_battery / 1e6, 2), "MNOK")
print("Surplus from base production: ", round(surplus_base / 1e6, 2), "MNOK")
print("Total surplus: ", round(pyo.value(model.objective) / 1e6, 2), "MNOK")
print(f'Optmial TES size: {round(model.mass_TES.value, 1)} kg')
print(f'TES storage capacity: {round(pyo.value(model.u_TES_max), 2)} kWh')

# %% ----- Plotting results ----- #

hourslist = list(hours)
massflow_base = list(pyo.value(model.massflow_base[hour]) for hour in hours)
massflow_out = list(pyo.value(model.massflow_out[hour]) for hour in hours)
massflow_in = list(pyo.value(model.massflow_in[hour]) for hour in hours)
u = list(pyo.value(model.u_TES[hour]) for hour in hours)
power_TES = list(pyo.value(model.power_TES[hour]) for hour in hours)
power_base = list(pyo.value(model.power_base[hour]) for hour in hours)
battery_energy = list((pyo.value(model.u_TES[hour])) for hour in hours)
m_in = list((pyo.value(model.massflow_in[hour])) for hour in hours)
m_out = list((pyo.value(model.massflow_out[hour])) for hour in hours)
m_base = list((pyo.value(model.massflow_base[hour])) for hour in hours)
total_power = [power_TES[i] + power_base[i] for i in hours]


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
        plt.ylabel("Power [kW]")
        plt.title(f"Power output for month number {i}")
        plt.legend()
        plt.show()


def plot_power_output_sorted():
    power_TES.sort(reverse = True)
    power_base.sort(reverse = True)
    plt.plot(hourslist, power_TES, color = "hotpink", label = "power from battery storage")
    plt.plot(hourslist, power_base, color = "mediumpurple", label = "base power")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kW]")
    plt.title(f"Power output through the year")
    plt.legend()
    plt.show()


def plot_total_power_output_sorted():
    total_power.sort(reverse = True)
    plt.plot(hourslist, total_power, color = "lightseagreen", label = "Total power outbut [MW]")
    plt.fill_between(hourslist, total_power, color = 'palegreen', alpha = 0.4)
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kW]")
    plt.title(f"Total power output through the year")
    plt.legend()
    plt.show()

def plot_battery_storage():
    plt.plot(hourslist, battery_energy, color = 'dodgerblue')
    plt.fill_between(hourslist, battery_energy, color = 'skyblue', alpha = 0.4)
    plt.title("Stored energy in TES through the year")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kWh]")
    plt.show()

def plot_battery_storage_sorted():
    battery_energy.sort(reverse = True)
    plt.plot(hourslist, battery_energy, color = 'dodgerblue')
    plt.fill_between(hourslist, battery_energy, color = 'skyblue', alpha = 0.4)
    plt.title("Stored energy in TES through the year")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kWh]")
    plt.show()

plot_battery_storage_sorted()
#plot_battery_storage()
plot_power_output_sorted()

# TODO: husk at batteriet har en "duration" siden varme ikke kan lagres særlig lenge