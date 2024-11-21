import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from Data_handeling import read_previous_spot
import math
"""Construct a thermal energy storage pyomo block for optimisation."""

print(f'- CONSTRUCTING THERMAL ENERGY STORAGE')

# Calculate the enthalpy of each steam flow
# ---------------------------------------------

# Create pyomo block
blk = pyo.Block()

# Number of modules
blk.n_mods = pyo.Param(initialize = 3, doc = '[-]')
# blk.n_mods            = pyo.Var(bounds=(1,5), doc='[-]')

step1 = 0
tf = 8759
hour = range(step1, tf + 1)
hour = pyo.Set(initialize=range(step1, tf + 1))

# Mass flows
# -------------------------
blk.M_steam_in = pyo.Var(hour, bounds = (0, 173.5 * blk.n_mods), doc = '[kg/s]')  # 173.5*blk.n_mods
blk.M_steam_in_state = pyo.Var(hour, within = pyo.Binary)
blk.M_steam_out = pyo.Var(hour, bounds = (0, 173.5 * blk.n_mods), doc = '[kg/s]')  # 173.5*blk.n_mods

# Heat levels and flows
# -------------------------
blk.Q_storage_cap = pyo.Param(initialize = 1317.91, doc = '[GJ]')
blk.Q_storage = pyo.Var(hour, bounds = (0, None), doc = '[GJ]')
blk.Q_to_tes = pyo.Var(hour, bounds = (0, None), doc = '[GJ]')
blk.Q_from_tes = pyo.Var(hour, bounds = (0, None), doc = '[GJ]')
blk.Q_loss = pyo.Var(hour, bounds = (0, None), doc = '[GJ]')

# Cost elements
# ---------------------
blk.build_cost = pyo.Var(bounds = (0, None), doc = '[EUR]')
blk.op_cost = pyo.Var(bounds = (0, None), doc = '[EUR]')

C_build = 28  # [EUR/kWh]


# Build cost
def build_costs_tes_rule(_blk):
    return _blk.build_cost == C_build * _blk.Q_storage_cap * _blk.n_mods * 0.000277 * 10**6  # converting from GJ to kWh


blk.build_costs_tes = pyo.Constraint(rule = build_costs_tes_rule)


# Operation cost 
def operation_costs_tes_rule(_blk):
    return _blk.op_cost == 0.02 * C_build * _blk.Q_storage_cap * _blk.n_mods * 0.000277 * 10**6  # converting from GJ to kWh


blk.operation_costs_tes = pyo.Constraint(rule = operation_costs_tes_rule)

U = 0.1939 * 70  # W/m2K


# Heat loss 
def heat_loss_rule(_blk, t):
    return _blk.Q_loss[t] == U * (((_blk.Q_storage_cap * 1e6 / (212 * 2200 * math.pi))**(2 / 3)) * 3 * math.pi) * (
                282 - 11) * 3600 / 1e9 * _blk.n_mods  # calculating the heat loss U * A * (T_storage - T_amb) - A based on a cylindrical design with h = r


blk.heat_loss = pyo.Constraint(hour, rule = heat_loss_rule)


# Storage capacity constraint
def cap_rule(_blk, t):
    return _blk.Q_storage[t] <= _blk.Q_storage_cap * _blk.n_mods


blk.cap = pyo.Constraint(hour, rule = cap_rule)


# Heat transfer constraints
def heat_to_tes_rule(_blk, t):
    return _blk.Q_to_tes[t] == (_blk.M_steam_in[t] * (60**2) * (2760 - 650)) / (10**6)


blk.heat_to_tes = pyo.Constraint(hour, rule = heat_to_tes_rule)


def heat_from_tes_rule(_blk, t):
    return _blk.Q_from_tes[t] == (_blk.M_steam_out[t] * (60**2) * (2774 - 1000)) / (10**6)


blk.heat_from_tes = pyo.Constraint(hour, rule = heat_from_tes_rule)


# Steam mass balance
def tes_heat_balance_rule(_blk, t):
    if t == 0:
        return _blk.Q_storage[t] == 0
    else:
        return _blk.Q_storage[t] == _blk.Q_storage[t - 1] + _blk.Q_to_tes[t - 1] - _blk.Q_from_tes[t - 1] - _blk.Q_loss[
            t - 1]


blk.tes_heat_balance = pyo.Constraint(hour, rule = tes_heat_balance_rule)

"""Extract the results from the block."""
index_hours = list(blk.Q_storage.keys())
Q_storage = []
M_steam_in = []
M_steam_out = []
Q_to_tes = []
Q_from_tes = []
Q_loss = []
for h in index_hours:
    Q_storage[h]        = blk.Q_storage[h].value
    M_steam_in[h]       = blk.M_steam_in[h].value
    M_steam_out[h]      = blk.M_steam_out[h].value
    Q_to_tes[h]         = blk.Q_to_tes[h].value
    Q_from_tes[h]       = blk.Q_from_tes[h].value
    Q_loss[h]           = blk.Q_loss[h].value
    
    
plt.plot(hour, Q_storage)
plt.show()
