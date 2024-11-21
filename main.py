'''
This is the main
'''
import Siste as opt
import Data_handeling as dh
import Plots as plot

''' -------------------------- User Interface --------------------------'''

# Year to examine (2023 or 2050)
year = 2023

# Zone within Norway to examine
zone = "NO1"

# Marginal cost of production NOK/kWh (marginal cost)
MC_base = 0.20
MC_TES = MC_base

# Interesrt rate
interest_rate = 0.045

# TES material cost per kg
investment_cost_weight = 5  # NOK/kg

# Peaking plant cost per kW
investment_cost_cap = (12 + 234 + 1142) * 10  # NOK/kW

# Lifetime of TES
lifetime = 60

def main():

    # ---------------------------- Plot spot prices ---------------------------- #
    def data_handling_output():
        # Plot spot prices
        dh.plot_spot(year = 2050, zone = 'NO1')

    # ---------------- Solve sizing problem for chosen scenario ---------------- #

    def Solve_problem():
        # Solve optimization model
        model = opt.run_model(year, zone, MC_base, MC_TES, interest_rate, investment_cost_weight, investment_cost_cap, lifetime)

        # Print results
        opt.print_res(model, year, zone, MC_TES, MC_base)

        # Make lists with results
        lists= plot.get_lists(model)

        # Plot lists
        plot.plot_battery_storage_sorted(lists)
        plot.plot_power_output_sorted(lists)


    # Run all tasks (comment out the tasks you don't want to run)
    #data_handling_output()
    Solve_problem()

if __name__ == "__main__":
    main()




