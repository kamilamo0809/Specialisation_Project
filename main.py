'''
This is the user interface and main. Change variables under -- User Interface -- and then run
the code to obtain the optimal size for thew problem. The optimal size of the TES will be chosen
to the size with the most net benefit

Default variables (used in the project analysis) are noted to the right

Made by Kamilla Aarflot Moen
course: TET4510
'''

import Sizing_of_TES_Optmiization as opt
import Data_handeling as dh
import Plots as plot
import pandas as pd


''' -------------------------- User Interface --------------------------'''

# Year to examine (2023 or 2040)
year = 2040
zone = "NO1"
start, stop = (0, 8759)

# Marginal cost of production NOK/kWh (marginal cost)
MC_base = 0.20                      # NOK/kWh
MC_TES = MC_base                    # NOK/kWh

# Interest rate
interest_rate = 0.045

# lifetime of TES
lifetime = 20                       # years

# TES material cost per kg
inv_weight = 6                      # NOK/kg
inv_tank = 37240                    # NOK/m3
inv_peak = 15500                    # NOK/kW

charging_time = 12                  # h
discharging_time = 12               # h

# Tank properties
U_value = 0.0133                    # kJ/m2K
temperature_ambient = 11 + 273.15   # K

# Material properties
density = 1870                      # kg/m3
temperature_max = 565 + 273.15      # K
temperature_min = 265 + 273.15      # K
specific_heat = 1.600               # kJ / kg * K

# Charge and discharge enthalpies
h_charge_in = 2760                  # kJ/kg
h_charge_out = 650                  # kJ/kg
h_discharge_in = 2774               # kJ/kg
h_discharge_out = 1000              # kJ/kg

# Max size on TES system
max_mass_TES = 80000000             # kg

# Enthalpies
h_turbine_in = 3100                 # kJ/kg
h_turbine_out = 2200                # kJ/kg

# Efficiencies
efficiency_turbine = 1
efficiency_charge = 1
efficiency_discharge = 1

# Max steam generation from reactor
reactor_power_max = 300 * 1000      # kW

# Ramping rate
ramping_rate = 5                    # %/min



def main():

    # ---------------------------- Plot spot prices ---------------------------- #
    def data_handling_output():
        # Plot spot prices
        dh.plot_spot(year = year, zone = zone)

    # ---------------- Solve sizing problem for chosen scenario ---------------- #
    tes = True
    if max_mass_TES == 0:
        tes = False

    def Solve_problem():
        # Solve optimization model

        model, results = opt.run_model(year, zone, charging_time, discharging_time, MC_base, MC_TES, interest_rate, inv_weight,
              inv_tank, inv_peak, lifetime, start, stop, U_value, density, temperature_ambient, temperature_max, temperature_min, specific_heat, h_turbine_in, h_turbine_out, h_discharge_in,
              h_discharge_out, h_charge_in, h_charge_out, max_mass_TES, efficiency_turbine, efficiency_charge, efficiency_discharge, reactor_power_max, ramping_rate)

        # Print results and save to csv
        opt.print_res(model, year, zone, MC_TES, MC_base, start, stop, density)

        result_dict = opt.get_dict(model, f'res{year}{tes}mfb.csv', start, stop)

    # ---------------- Plow results from optimization problem ---------------- #
    def Plot_problem():
        # Get results
        result_dict = pd.read_csv(f'res{year}{tes}.csv').to_dict(orient='dict')

        plot.plot_everything_month([5, 6, 7], result_dict, year, zone)
        plot.plot_everything_hour(result_dict)
        plot.duration(result_dict)
        plot.plot_total_power(result_dict)
        plot.plot_efficiency()
        plot.plot_material_bar()
        plot.plot_everything_month_aggregated(result_dict, year, zone)
        plot.plot_CF()
        plot.plot_VF()
        plot.plot_diff_VF_and_CF()

    # Run all tasks (comment out the tasks you don't want to run)
    data_handling_output()
    Solve_problem()
    Plot_problem()



if __name__ == "__main__":
    main()




