'''
This file contains the TES sizing optimization problem.

run_model: solves the problem with the given parameters
print_res: prints results and key values from the the input model
get_lists: returns lists of variable values for all hours 

Made by Kamilla Aarflot Moen
course: TET4510
'''


''' ----- Dependencies ----- '''

import numpy as np
import pyomo.environ as pyo
import pandas as pd
from pyomo.opt import SolverFactory
from Data_handeling import read_previous_spot
from sty import fg

def run_model(year, zone, charging_time, discharging_time, MC_base, MC_TES, interest_rate, inv_weight,
              inv_tank, inv_peak, lifetime, start, stop, U, density, temp_amb, temp_max, temp_min, spec_heat, h_turb_in, h_turb_out, h_dis_in,
              h_dis_out, h_ch_in, h_ch_out, max_mass_TES, eff_turb, eff_ch, eff_dis, power_max, ramp):
    '''
    This function sets up and solves the sizing of TES optimization problem
    :param year: year to analyse
    :param zone: power market zone within norway
    :param charging_time: time to fully charge the TES with full power
    :param discharging_time: time to fully discharge the TES with full power
    :param MC_base: Marginal cost of production by base plant
    :param MC_TES: Marginal cost of production by peaking plant
    :param interest_rate: expected interest rate for the comming years
    :param inv_weight: The cost of 1 unit TES material
    :param inv_tank: The cost of the tank per volume
    :param inv_peak: The cost of the peaking plant per power capacity unit
    :param lifetime: The expected lifespan of the TES
    :param start: hour number
    :param stop: hour number
    :param U: Heat transfer coefficient of tank
    :param density: of tes material
    :param temp_amb: Outside the tank
    :param temp_max: Inside the tank
    :param temp_min: Inside the tank
    :param spec_heat: Of TES material
    :param h_turb_in: Enthalpy at the inlet of the turbine
    :param h_turb_out: Enthalpy at the outlet of the turbine
    :param h_dis_in: Enthalpy at the discharge inlet
    :param h_dis_out: Enthalpy at the discharge outlet
    :param h_ch_in: Enthalpy at the charge inlet
    :param h_ch_out: Enthalpy at the charge inlet
    :param max_mass_TES: The maximum mass of the TES material
    :param eff_turb: Turbine efficiency
    :param eff_ch: Charging efficiency
    :param eff_dis: Discharge efficiency
    :param power_max: Maximum power from the reactor
    :param ramp: %/min the of power output change in the generator

    :return: pyomo model
    '''

    ''' --------------------------- Model setup ------------------------- '''

    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)

    '''------------ Reading spot price data from csv or excel------------ '''

    power_prices = read_previous_spot(year, zone)

    ''' ------------------------------ Set ------------------------------ '''
    model.time = pyo.Set(initialize=range(start, stop + 1))

    ''' --------------------------- Parameters -------------------------- '''

    # Initial internal energy
    u_TES_init                  = 0                                 # kWh

    h_delta_charge              = h_ch_in - h_ch_out                # kJ/kg
    h_delta_discharge           = h_dis_in - h_dis_out              # kJ/kg
    h_delta_turbine             = h_turb_in - h_turb_out            # kJ/kg

    massflow_gen_max            = power_max / h_delta_turbine       # kg/s

    ramping                     = ramp * 0.6 * massflow_gen_max     # kg/h

    # Parameters
    model.h_delta_charge        = pyo.Param(initialize = h_delta_charge)
    model.h_delta_discharge     = pyo.Param(initialize = h_delta_discharge)
    model.h_delta_turbine       = pyo.Param(initialize = h_delta_turbine)
    model.massflow_gen_max      = pyo.Param(initialize = massflow_gen_max)
    model.ramping               = pyo.Param(initialize = ramping)
    model.lifetime              = pyo.Param(initialize = lifetime)


    ''' --------------------------- Variables --------------------------- '''

    # TES size (kW)
    model.mass_TES         = pyo.Var(within = pyo.NonNegativeReals)                          # kg
    model.u_TES_max        = pyo.Var(within = pyo.NonNegativeReals)                          # kWh
    model.peak_plant_cap   = pyo.Var(within = pyo.NonNegativeReals)                          # kW
    model.max_charge       = pyo.Var(within = pyo.NonNegativeReals)

    # Charging (1 or 0)
    model.is_charging      = pyo.Var(model.time, domain = pyo.Binary)

    # Internal energy TES (kJ)
    model.u_TES            = pyo.Var(model.time, within = pyo.NonNegativeReals)      # kWh

    # Mass flows (kg/s)
    model.massflow_base    =   pyo.Var(model.time, within = pyo.NonNegativeReals)    #kg/s
    model.massflow_in      =     pyo.Var(model.time, within = pyo.NonNegativeReals)  #kg/s
    model.massflow_out     =    pyo.Var(model.time, within = pyo.NonNegativeReals)   #kg/s
    model.massflow_gen     =    pyo.Var(model.time, within = pyo.NonNegativeReals)   #kg/s

    # Power (kW)
    model.power_base       = pyo.Var(model.time, within = pyo.NonNegativeReals)      # kW
    model.power_peak       = pyo.Var(model.time, within = pyo.NonNegativeReals)      # kW

    # Heat flows (kJ)
    model.q_in             = pyo.Var(model.time, within = pyo.NonNegativeReals)      # kW / kJ/s
    model.q_out            = pyo.Var(model.time, within = pyo.NonNegativeReals)      # kW / kJ/s
    model.q_loss           = pyo.Var(model.time, within = pyo.NonNegativeReals)      # kW / kJ/s

    # TES tank surface
    model.A                = pyo.Var(within = pyo.NonNegativeReals)                           # m2

    model.temp             = pyo.Var(model.time, within = pyo.NonNegativeReals)      # K


    ''' --------------------------- Constraints --------------------------- '''

    # Base power generation                     (kW = kg/s * kJ/kg = kJ/s)
    def base_power_rule(model, hour):
        return model.power_base[hour] == model.massflow_base[hour] * model.h_delta_turbine * eff_turb   # kW / kJ/s

    # peak power generation                     (kW = kg/s * kJ/kg = kJ/s)
    def TES_power_rule(model, hour):
        return model.power_peak[hour] == model.massflow_out[hour] * model.h_delta_turbine * eff_turb    # kW / kJ/s

    # Energy balance in TES                     (kWh = kW * 1h)
    def energy_balance_rule(model, hour):
        if hour == start or hour == stop:
            return model.u_TES[hour] == u_TES_init

        else:
            return (model.u_TES[hour] ==    model.u_TES[hour - 1]
                                           + (model.q_in[hour - 1]
                                           - model.q_out[hour - 1]
                                           - model.q_loss[hour - 1]))

    # Heat flow in                              (kW = kg/s * kJ/kg = kJ/s)
    def heat_flow_in_rule(model, hour):
        return model.q_in[hour] == model.massflow_in[hour] * model.h_delta_charge * eff_ch

    # Heat flow out                             (kW = kg/s * kJ/kg = kJ/s)
    def heat_flow_out_rule(model, hour):
        return model.q_out[hour] == (model.massflow_out[hour] * model.h_delta_discharge)/eff_dis

    # Heat loss                                 Q_loss = U * A * (T_tank - T_ambient)
    def heat_loss_rule(model, hour):            # kW = kW/m2*K * m2 * K = kJ/s
        return model.q_loss[hour] == U * model.A * (model.u_TES[hour] / ((spec_heat / 3600) * 63300000))

    # Choose surface based on the mass of the TES material
    def surface_rule(model):
        return model.A >= 3 * np.pi * (63300000 / (np.pi * density))**(2/3)

    # Charge capacity rule                      (kW * 1h = kWh)
    def charge_power_cap_rule(model, hour): # kg/s <= kJ
        return model.q_in[hour] <= (model.u_TES_max /charging_time) * model.is_charging[hour]

    # Disharge capacity rule                    (kW * 1h = kWh)
    def discharge_power_cap_rule(model, hour):
        return model.q_out[hour] <= (model.u_TES_max /discharging_time) * (1 - model.is_charging[hour])

    # Max TES capacity
    def max_TES_cap_rule(model, hour):
        return (model.u_TES[hour] <= model.u_TES_max)

    def max_TES(model):
        return (model.u_TES_max == model.mass_TES * spec_heat * (temp_max - temp_min) / 3600)

    # ramping power min
    def min_ramping_rule(model, hour):
        if hour == start:
            return pyo.Constraint.Skip
        else:
            return -model.ramping <= model.massflow_gen[hour] - model.massflow_gen[hour - 1]

    # ramping power max
    def max_ramping_rule(model, hour):
        if hour == start:
            return pyo.Constraint.Skip
        else:
            return model.massflow_gen[hour] - model.massflow_gen[hour - 1] <= model.ramping

    # Max reactor generation
    def max_generation_rule(model, hour):
        return model.massflow_gen[hour] <= model.massflow_gen_max

    # Max mass in TES
    def max_size_rule(model):
        return model.mass_TES <= max_mass_TES

    # mass conservation
    def mass_conservation_rule(model, hour):
        return (model.massflow_gen[hour] == model.massflow_in[hour] + model.massflow_base[hour])

    # Peak plant size must be greater than the maximum dispatched power
    def greatest_output_rule(model, hour):
        return model.peak_plant_cap >= model.power_peak[hour]

    # TES size must be equal to the maximum charge energy
    def link_size_and_power(model):
        return model.u_TES_max == charging_time * model.max_charge

    # Maximum charge power must be less than the mass flow
    def max_charge_rule(model):
        return model.max_charge <= model.massflow_gen_max * model.h_delta_charge * eff_ch


    ''' ----- Objective Function: Maximize the surplus (profit) ----- '''

    def objective_rule(model):
        # Maximize net revenue (sizing problem)
        return (- (inv_weight * model.mass_TES + model.peak_plant_cap * inv_peak + model.mass_TES/density * inv_tank) * (interest_rate / (1 - (1 + interest_rate) ** (- model.lifetime)))
                + sum(
                    (power_prices[hour] - MC_base ) * model.power_base[hour] +
                    (power_prices[hour] - MC_TES ) * model.power_peak[hour]
                    for hour in model.time))


        # Maximize income
        # return sum(
        #             (power_prices[hour] - MC_base ) * model.power_base[hour] +
        #             (power_prices[hour] - MC_TES ) * model.power_peak[hour]
        #             for hour in model.time)

    ''' -------------------------------------------------------------'''

    # Constraints
    model.base_power                = pyo.Constraint(model.time, rule = base_power_rule)
    model.TES_power                 = pyo.Constraint(model.time, rule = TES_power_rule)
    model.energy_balance            = pyo.Constraint(model.time, rule = energy_balance_rule)
    model.charge_power_cap_rule     = pyo.Constraint(model.time, rule = charge_power_cap_rule)
    model.discharge_power_cap_rule  = pyo.Constraint(model.time, rule = discharge_power_cap_rule)
    model.max_TES_cap               = pyo.Constraint(model.time, rule = max_TES_cap_rule)
    model.max_TES                   = pyo.Constraint(rule = max_TES)
    model.min_ramping               = pyo.Constraint(model.time, rule = min_ramping_rule)
    model.max_ramping               = pyo.Constraint(model.time, rule = max_ramping_rule)
    model.max_generation            = pyo.Constraint(model.time, rule = max_generation_rule)
    model.max_size                  = pyo.Constraint(rule = max_size_rule)
    model.mass_conservaton          = pyo.Constraint(model.time, rule = mass_conservation_rule)
    model.greatest_output           = pyo.Constraint(model.time, rule = greatest_output_rule)
    model.heat_flow_in              = pyo.Constraint(model.time, rule = heat_flow_in_rule)
    model.heat_flow_out             = pyo.Constraint(model.time, rule = heat_flow_out_rule)
    model.heat_loss                 = pyo.Constraint(model.time, rule = heat_loss_rule)
    model.surface                   = pyo.Constraint(rule = surface_rule)
    model.max_charge_rule           = pyo.Constraint(rule = max_charge_rule)
    model.link_const                = pyo.Constraint( rule =link_size_and_power)

    # Objective
    model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)


    ''' -------------------- Solving the optimization problem ----------------------- '''

    opt = SolverFactory("gurobi", solver_io="python")

    # Set a time limit for the solver (in seconds)
    opt.options['TimeLimit'] = 500

    # Set a MIP gap tolerance for stopping criteria
    opt.options['MIPGap'] = 0.013

    results = opt.solve(model, load_solutions = True, tee = True)

    print('\n')

    # Print solver status
    if results.solver.status == pyo.SolverStatus.ok:
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print(fg(51) + "Optimal solution found.")
        else:
            print("Solver stopped with termination condition:", results.solver.termination_condition)
    else:
        print("Solver error:", results.solver.status)

    return model, results

''' ----- Printing results ----- '''

def print_res(model, year, zone, MC_TES, MC_base, start, stop, density):
    '''
    This function prints the result from the optimized model nicely

    - Objective value
    - Surplus from battery and base production
    - Investment cost
    - Optimal TES size
    - Storage capacity
    - Peaking plant size
    - Max charge power
    - Tank surface areal
    - Tank volume
    '''

    # Read power prices for given year and zone
    power_prices = read_previous_spot(year, zone)

    # Define the list of all hours
    hours = range(start, stop + 1)

    # Calculate surplus and investment cost
    surplus_battery = sum((pyo.value(power_prices[hour]) - MC_TES) * pyo.value(model.power_peak[hour]) for hour in hours)
    surplus_base = sum((pyo.value(power_prices[hour]) - MC_base) * pyo.value(model.power_base[hour]) for hour in hours)
    investment = (- pyo.value(model.objective) + surplus_battery + surplus_base)

    # Print
    print(fg(214) + '------------------------------------ RESULTS ------------------------------------' + fg.rs)
    print(fg(215) + f"Objective value: {round(pyo.value(model.objective) / 1e6, 4)} MNOK \n" + fg.rs)

    print(fg(216) + f"Surplus from battery storage: {round(surplus_battery / 1e6, 4)} MNOK \n"
                   f"Surplus from base production: {round(surplus_base / 1e6, 4)} MNOK \n"
                   f"Cost of investments: {round(investment / 1e6, 4)} MNOK \n" + fg.rs)

    print(fg(217) + f'Optimal TES size: {round(model.mass_TES.value, 5)} kg\n'
                    f'TES storage capacity: {round(pyo.value(model.u_TES_max) / 1000, 2)} MWh\n' + fg.rs)


    print(fg(218) + f'Peak plant size = {round(pyo.value(model.peak_plant_cap) / 1000, 2)} MW\n' + fg.rs)

    print(fg(219) + f'Other variables: \n'
                   f'Max charge power = {round(pyo.value(model.max_charge)/1000, 2)} MW\n'
                   f'TES tank surface areal = {round(pyo.value(model.A), 2)} m2\n'
                    f'TES tank volume = {round(pyo.value(model.mass_TES) / density, 2)} m3' + fg.rs)


def get_dict(model, start = 0, stop = 8759):
    '''
    This function makes lists out of pyomo variables from a solved model for each timestamp

    return: dictionary og all values
    '''
    hours = range(start, stop + 1)
    power_peak = list(pyo.value(model.power_peak[hour] / 1000) for hour in hours)
    power_base = list(pyo.value(model.power_base[hour] / 1000) for hour in hours)

    everything = {'time':           range(start, stop + 1),                                             # h
                  'mass flow base': list(pyo.value(model.massflow_base[hour]) for hour in hours),       # kg/s
                  'mass flow out':  list(pyo.value(model.massflow_out[hour]) for hour in hours),        # kg/s
                  'mass flow in':   list(pyo.value(model.massflow_in[hour]) for hour in hours),         # kg/s
                  'power peak':     list(pyo.value(model.power_peak[hour] / 1000) for hour in hours),   # MW
                  'power base':     list(pyo.value(model.power_base[hour] / 1000) for hour in hours),   # MW
                  'battery':        list(pyo.value(model.u_TES[hour] / 1000) for hour in hours),        # MWh
                  'power total':    [(power_peak[i] + power_base[i]) for i in hours],                   # MW
                  'loss':           list(pyo.value(model.q_loss[hour] / 1000) for hour in hours),       # MW
                  'heat in':        list(pyo.value(model.q_in[hour] / 1000) for hour in hours),         # MW
                  'heat out':       list(pyo.value(model.q_out[hour] / 1000) for hour in hours)}        # MW

    df = pd.DataFrame(everything)
    df.to_csv('Results.csv', index = False)

    return everything

def get_objective(model):
    return pyo.value(model.objective)