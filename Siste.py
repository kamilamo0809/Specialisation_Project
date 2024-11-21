''' ----- Dependencies ----- '''

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import pandas as pd
from Data_handeling import read_previous_spot
from tqdm import tqdm
import gurobipy as grb

def run_model(year = 2050, zone = 'NO1', MC_base = 0.2, MC_TES = 0.2, interest_rate = 0.045, investment_cost_weight = 5,
              investment_cost_cap = (12 + 234 + 1142) * 10, lifetime = 60, step1 = 0, tf = 8759):

    annuity = interest_rate / (1 - (1 + interest_rate) ** (- lifetime))

    ''' ----- Model setup ----- '''

    # Create a concrete Pyomo model
    model = pyo.ConcreteModel()

    '''----- Reading spot price data from csv or excel----- '''

    power_prices = read_previous_spot(year, zone)

    '''  ----- Set ----- '''
    model.time = pyo.Set(initialize=range(step1, tf + 1))

    ''' ----- Parameters ----- '''

    # TODO: find enthalpies, efficiencies, temperatures
    # Max size on TES system
    max_mass_TES = 10000 # TODO: sett riktig her

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
    efficiency_turbine = 1 # TODO: sett tall her
    efficiency_charge = 1
    efficiency_discharge = 1

    # TES temperatures
    temperature_max = 565 + 273.15      # K
    temperature_min = 265 + 273.15      # K

    # Specific heat capacity
    specific_heat = 1600                 # kJ / kg * K

    # Max steam generation from reactor
    massflow_gen_max = 300000 / h_delta_turbine  # kg/s

    # Ramping
    pro = 1 # %/min
    ramping = pro * 0.6 * massflow_gen_max  # kg/h

    charge_cap = 200000 / h_delta_turbine # kg/s # TODO finn ut hva charge og discharge power er
    discharge_cap = 200000 / h_delta_turbine # kg/s


    ''' ----- Variables ----- '''

    # TES size (kW)
    model.mass_TES = pyo.Var(within = pyo.NonNegativeReals)             # kg
    model.u_TES_max = pyo.Var(within = pyo.NonNegativeReals)            # kJ
    model.max_peak_power = pyo.Var(within = pyo.NonNegativeReals)       # kW

    # Charging (1 or 0)
    model.is_charging = pyo.Var(model.time, domain = pyo.Binary)

    # Internal energy TES (kJ)
    model.u_TES = pyo.Var(model.time, within = pyo.NonNegativeReals)

    # Mass flows (kg/s)
    model.massflow_base =   pyo.Var(model.time, within = pyo.NonNegativeReals)
    model.massflow_in =     pyo.Var(model.time, within = pyo.NonNegativeReals)
    model.massflow_out =    pyo.Var(model.time, within = pyo.NonNegativeReals)
    model.massflow_gen =    pyo.Var(model.time, within = pyo.NonNegativeReals)

    # Power (kW)
    model.power_base = pyo.Var(model.time, within = pyo.NonNegativeReals)
    model.power_peak = pyo.Var(model.time, within = pyo.NonNegativeReals)


    ''' ----- Constraints ----- '''

    # Base power generation
    def base_power_rule(model, hour):
        return model.power_base[hour] == model.massflow_base[hour] * h_delta_turbine

    # peak power generation
    def TES_power_rule(model, hour):
        return model.power_peak[hour] == model.massflow_out[hour] * h_delta_turbine

    # Energy balance in TES # TODO: endre denne
    def energy_balance_rule(model, hour):
        if hour == step1:
            return model.u_TES[hour] == u_TES_init + model.massflow_in[hour] * h_delta_charge * efficiency_charge

        else:
            return (model.u_TES[hour] ==    model.u_TES[hour - 1]
                                           + model.massflow_in[hour] * h_delta_charge * efficiency_charge
                                           - (model.massflow_out[hour] * h_delta_discharge)/efficiency_discharge)

    # End internal energy
    def internal_energy_end_rule(model):
        return model.u_TES[tf] == u_TES_init

    # Can not discharge the first hour
    def discharge_first_hour(model):
        return model.massflow_out[step1] == 0

    # Can not charge and discharge at the same time
    def charge_power_cap_rule(model, hour):
        return model.massflow_in[hour] <= charge_cap * model.is_charging[hour]

    def discharge_power_cap_rule(model, hour):
        return model.massflow_out[hour] <= discharge_cap * (1 - model.is_charging[hour])


    # Max TES capacity
    def max_TES_cap_rule(model, hour):
        return (model.u_TES[hour] <= model.u_TES_max)

    def max_TES(model):
        return (model.u_TES_max == model.mass_TES * specific_heat * (temperature_max - temperature_min) / 3600) # TODO: finn ut om det skal vare * 3600 her


    # ramping power min
    def min_ramping_rule(model, hour):
        if hour == step1:
            return pyo.Constraint.Skip
        else:
            return -ramping <= model.massflow_gen[hour] - model.massflow_gen[hour - 1]


    # ramping power max
    def max_ramping_rule(model, hour):
        if hour == step1:
            return pyo.Constraint.Skip
        else:
            return model.massflow_gen[hour] - model.massflow_gen[hour - 1] <= ramping


    # Max reactor generation
    def max_generation_rule(model, hour):
        return model.massflow_gen[hour] <= massflow_gen_max


    # Max mass for TES
    def max_size_rule(model):
        return model.mass_TES <= max_mass_TES


    # mass conservation
    def mass_conservation_rule(model, hour):
        return (model.massflow_gen[hour] == model.massflow_in[hour] + model.massflow_base[hour])

    # Greatest output power from peaking turbine
    def greatest_output_rule(model, hour):
        return model.max_peak_power >= model.power_peak[hour]

    ''' ----- Objective Function: Maximize the surplus (profit) ----- '''


    def objective_rule(model):
        return (- investment_cost_weight * model.mass_TES * annuity - model.max_peak_power * investment_cost_cap  * annuity
               + sum(( power_prices[hour] - MC_base ) * model.power_base[hour]
               + ( power_prices[hour] - MC_TES ) * model.power_peak[hour] for hour in model.time)) / 1000


    ''' -------------------------------------------------------------'''

    # Constraints
    model.base_power = pyo.Constraint(model.time, rule = base_power_rule)
    model.TES_power = pyo.Constraint(model.time, rule = TES_power_rule)
    model.energy_balance = pyo.Constraint(model.time, rule = energy_balance_rule)
    model.internal_energy_end = pyo.Constraint(rule = internal_energy_end_rule)
    model.discharge_first_hour = pyo.Constraint(rule = discharge_first_hour)
    model.charge_power_cap_rule = pyo.Constraint(model.time, rule = charge_power_cap_rule)
    model.discharge_power_cap_rule = pyo.Constraint(model.time, rule = discharge_power_cap_rule)
    model.max_TES_cap = pyo.Constraint(model.time, rule = max_TES_cap_rule)
    model.max_TES = pyo.Constraint(rule = max_TES)
    model.min_ramping = pyo.Constraint(model.time, rule = min_ramping_rule)
    model.max_ramping = pyo.Constraint(model.time, rule = max_ramping_rule)
    model.max_generation = pyo.Constraint(model.time, rule = max_generation_rule)
    model.max_size = pyo.Constraint(rule = max_size_rule)
    model.mass_conservaton = pyo.Constraint(model.time, rule = mass_conservation_rule)
    model.greatest_output = pyo.Constraint(model.time, rule = greatest_output_rule)

    # Objective
    model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)


    ''' ----- Solving the optimization problem ----- '''

    opt = SolverFactory("gurobi", solver_io="python")
    opt.options['TimeLimit'] = 600  # Optional: Set a time limit for the solver (in seconds)
    opt.options['MIPGap'] = 0.01  # Optional: Set a MIP gap tolerance for stopping criteria
    results = opt.solve(model, load_solutions = True, tee = True)

    print('\n')

    if results.solver.status == pyo.SolverStatus.ok:
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("Optimal solution found.")
        else:
            print("Solver stopped with termination condition:", results.solver.termination_condition)
    else:
        print("Solver error:", results.solver.status)

    return model

''' ----- Printing results ----- '''

def print_res(model, year, zone, MC_TES, MC_base, step1 = 0, tf = 8759):

    power_prices = read_previous_spot(year, zone)
    hours = range(step1, tf + 1)

    surplus_battery = sum((pyo.value(power_prices[hour]) - MC_TES ) * pyo.value(model.power_peak[hour]) / 1000 for hour in hours)
    surplus_base = sum((pyo.value(power_prices[hour]) - MC_base ) * pyo.value(model.power_base[hour]) / 1000 for hour in hours)

    print("Surplus from battery storage: ", round(surplus_battery / 1e6, 4), "MNOK")
    print("Surplus from base production: ", round(surplus_base / 1e6, 4), "MNOK")
    print("Total surplus: ", round(pyo.value(model.objective) / 1e6, 3), "MNOK")
    print(f'Optmial TES size: {round(model.mass_TES.value, 1)} kg')
    print(f'TES storage capacity: {round(pyo.value(model.u_TES_max)/ 1000, 2)} MWh')

    geatest_output = 0
    for hour in hours:
        if pyo.value(model.power_peak[hour]) > geatest_output:
            geatest_output = pyo.value(model.power_peak[hour])

    print('Greatest power output from peak turbine = ', round(geatest_output / 1000, 2), ' MW')
