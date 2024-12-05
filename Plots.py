'''
This file plots battery output variables to compare and analyse results

Made by Kamilla Aarflot Moen
course: TET4510
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Data_handeling import read_previous_spot

specific_heat = 1.6

# Enthalpies
h_turbine_in = 3100  # kJ/kg
h_turbine_out = 2200  # kJ/kg

h_delta_turbine = h_turbine_in - h_turbine_out

h_charge_in = 2760  # kJ/kg
h_charge_out = 650  # kJ/kg

h_discharge_in = 2774  # kJ/kg
h_discharge_out = 1000  # kJ/kg

h_delta_charge = h_charge_in - h_charge_out
h_delta_discharge = h_discharge_in - h_discharge_out

''' ----- Plotting results ----- '''


def make_monthly_list(yearly_list):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    all_hours_in_year = []
    start_idx = 0
    for month in days_in_month:
        days = []
        all_hours_in_year.append(yearly_list[start_idx * 24: (start_idx + month) * 24])
        start_idx += month
    return all_hours_in_year


def daily_mean(monthly_list):
    daily_means = []
    for month in monthly_list:
        # Calculate the number of days in the month
        num_days = len(month) // 24

        # Divide each month's data into daily chunks (24 hours per day)
        days = [month[i * 24:(i + 1) * 24] for i in range(num_days)]

        # Calculate the mean for each day
        daily_means_month = [np.mean(day) for day in days]

        daily_means.append(daily_means_month)

    return daily_means


def plot_power_output(list_of_mont_numbers, results):
    hours = list(results['time'].values())
    power_peak = list(results['power peak'].values())
    power_base = list(results['power base'].values())

    time = make_monthly_list(hours)
    TES = make_monthly_list(power_peak)
    base = make_monthly_list(power_base)

    for i in list_of_mont_numbers:
        plt.plot(time[i - 1], TES[i - 1], color = "hotpink", label = "power from battery storage")
        plt.plot(time[i - 1], base[i - 1], color = "mediumpurple", label = "base power")
        plt.xlabel("Time [h]")
        plt.ylabel("Power [MW]")
        plt.title(f"Power output for month number {i}")
        plt.legend()
        plt.show()


def plot_total_power(results):
    hours = list(results['time'].values()).copy()
    power = list(results['power total'].values()).copy()

    # Create the plot with the first y-axis
    fig, ax1 = plt.subplots(figsize = (15, 5))

    # Plot power data with distinct colors and line styles for clarity
    plt.plot(hours, power, color = "darkturquoise", label = "Total power output [MW]", linewidth = 0.5)

    # Set the labels for the first y-axis (Power)
    plt.xlabel("Number of hour", fontsize = 10, labelpad = 10)
    plt.ylabel("Power [MW]", fontsize = 10, labelpad = 10)
    plt.title(f"Total dispatched power through the year", fontsize = 12, fontweight = 'bold', pad = 15)

    # Improve the appearance of legends
    plt.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3), fontsize = 10, frameon = False)

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_power_output_sorted(results):
    power_peak = list(results['power peak'].values()).copy()
    power_base = list(results['power base'].values()).copy()
    hourslist = list(results['time'].values())

    power_peak.sort(reverse = True)
    power_base.sort(reverse = True)

    plt.figure(figsize = (10, 6))  # Set figure size

    # Plot the data
    plt.plot(hourslist, power_peak, color = "darkorchid", linewidth = 2, label = "Peak Power")
    plt.plot(hourslist, power_base, color = "mediumturquoise", linewidth = 2, label = "Base Power")

    # Fill areas with better visual hierarchy
    plt.fill_between(hourslist, power_peak, power_base, color = "paleturquoise", alpha = 0.4, label = "Base Energy Output")
    plt.fill_between(hourslist, power_peak, color = "plum", alpha = 0.3, label = "Peak Energy Output")

    # Customize labels and title
    plt.xlabel("Time [h]", fontsize = 14)
    plt.ylabel("Power [MW]", fontsize = 14)
    plt.title("Power Dispatch Through The Year", fontsize = 16, fontweight = "bold")

    # Add grid for better readability
    plt.grid(color = "gray", linestyle = "--", linewidth = 0.5, alpha = 0.7)

    # Set legend outside plot to avoid overlap
    plt.legend(loc = "upper right", bbox_to_anchor = (1, 1), fontsize = 14, frameon = True)

    # Tighten layout for clean appearance
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_total_power_output_sorted(results):
    total_power = list(results['power total'].values()).copy()
    hourslist = list(results['time'].values())

    total_power.sort(reverse = True)
    plt.plot(hourslist, total_power, color = "lightseagreen", label = "Total power outbut [MW]")
    plt.fill_between(hourslist, total_power, color = 'palegreen', alpha = 0.4)
    plt.xlabel("Time [h]")
    plt.ylabel("Power [MW]")
    plt.title(f"Total power output through the year")
    plt.legend()
    plt.show()

def plot_battery_storage(results):
    hourslist = list(results['time'].values())
    battery_energy = list(results['battery'].values()).copy()

    plt.figure(figsize = (10, 6))  # Set figure size

    # Plot the data
    plt.plot(hourslist, battery_energy, color = 'dodgerblue', linewidth = 2, label = 'Stored energy (TES)')

    # Fill area below the curve
    plt.fill_between(hourslist, battery_energy, color = 'skyblue', alpha = 0.4, label = 'Energy stored')

    # Customize labels and title
    plt.xlabel("Time [h]", fontsize = 14)
    plt.ylabel("Energy [MWh]", fontsize = 14)
    plt.title("Stored Energy in TES Through the Year", fontsize = 16, fontweight = "bold")

    # Add grid for better readability
    plt.grid(color = "gray", linestyle = "--", linewidth = 0.5, alpha = 0.7)

    # Set legend outside plot to avoid overlap
    plt.legend(loc = "upper right", bbox_to_anchor = (1, 1), fontsize = 14, frameon = True)

    # Tighten layout for clean appearance
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_battery_storage_sorted(results):
    hourslist = list(results['time'].values())
    battery_energy = list(results['battery'].values()).copy()

    battery_energy.sort(reverse = True)

    plt.figure(figsize = (10, 6))  # Set figure size

    # Plot the data
    plt.plot(hourslist, battery_energy, color = 'dodgerblue', linewidth = 2, label = 'Stored energy (TES)')

    # Fill area below the curve
    plt.fill_between(hourslist, battery_energy, color = 'skyblue', alpha = 0.4, label = 'Energy stored')

    # Customize labels and title
    plt.xlabel("Time [h]", fontsize = 14)
    plt.ylabel("Energy [MWh]", fontsize = 14)
    plt.title("Stored Energy in TES Through the Year", fontsize = 16, fontweight = "bold")

    # Add grid for better readability
    plt.grid(color = "gray", linestyle = "--", linewidth = 0.5, alpha = 0.7)

    # Set legend outside plot to avoid overlap
    plt.legend(loc = "upper right", bbox_to_anchor = (1, 1), fontsize = 14, frameon = True)

    # Tighten layout for clean appearance
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_loss(results):
    loss = list(results['loss'].values()).copy()
    time = list(results['time'].values())

    loss.sort(reverse = True)

    plt.figure(figsize = (10, 6))  # Set figure size

    # Plot the data
    plt.plot(time, loss, color = 'dodgerblue', linewidth = 2, label = 'Stored energy (TES)')

    # Fill area below the curve
    plt.fill_between(time, loss, color = 'skyblue', alpha = 0.4, label = 'Energy loss')

    # Customize labels and title
    plt.xlabel("Time [h]", fontsize = 14)
    plt.ylabel("Energy loss [MWh]", fontsize = 14)
    plt.title("Lost energy through the year", fontsize = 16, fontweight = "bold")

    # Add grid for better readability
    plt.grid(color = "gray", linestyle = "--", linewidth = 0.5, alpha = 0.7)

    # Set legend outside plot to avoid overlap
    plt.legend(loc = "upper right", bbox_to_anchor = (1, 1), fontsize = 14, frameon = True)

    # Tighten layout for clean appearance
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_bat_storage_month(list_of_mont_numbers, results):
    hours = list(results['time'].values())
    bat_energy = list(results['battery'].values())

    time = make_monthly_list(hours)
    TES = make_monthly_list(bat_energy)

    for i in list_of_mont_numbers:
        plt.plot(time[i - 1], TES[i - 1], color = "hotpink", label = "stored energy")
        plt.xlabel("Time [h]")
        plt.ylabel("Power [MWh]")
        plt.title(f"Stored energy in TES for month number {i}")
        plt.legend()
        plt.show()



def plot_everything_month(list_of_mont_numbers, results, year, zone):
    # Extract data from the results
    hours = list(results['time'].values())
    power_peak = list(results['power peak'].values())
    power_base = list(results['power base'].values())
    tot = list(results['power total'].values())
    #pow_to_bat = list(results['heat in'].values())
    battery_energy = list(results['battery'].values())  # Energy in MWh
    power_prices = read_previous_spot(year, zone)
    power_prices = [power_prices[i] for i in power_prices]

    massflow_in = list(results['mass flow in'].values())
    pow_to_bat = [i * h_delta_turbine / 1000 for i in massflow_in]

    # Create monthly list from the yearly data
    time = make_monthly_list(hours)
    power_peak_monthly = make_monthly_list(power_peak)
    power_base_monthly = make_monthly_list(power_base)
    tot_monthly = make_monthly_list(tot)
    battery_energy_monthly = make_monthly_list(battery_energy)
    power_prices_monthly = make_monthly_list(power_prices)
    pow_to_bat_monthly = make_monthly_list(pow_to_bat)


    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    time_daily = [list(range(0, num)) for num in days_in_month]
    # Calculate daily means for power_peak and power_base
    power_peak_daily = daily_mean(power_peak_monthly)
    power_base_daily = daily_mean(power_base_monthly)
    tot_daily = daily_mean(tot_monthly)
    batt_en_daily = daily_mean(battery_energy_monthly)
    power_price_daily = daily_mean(power_prices_monthly)
    power_to_bat_daily = daily_mean(pow_to_bat_monthly)

    for i in list_of_mont_numbers:
        power_peak_data = power_peak_daily[i - 1]
        power_base_data = power_base_daily[i - 1]
        power_tot_data = tot_daily[i - 1]
        en_data = batt_en_daily[i-1]
        power_prices_data = power_price_daily[i - 1]
        power_to_bat = power_to_bat_daily[i-1]
        time = time_daily[i-1]

        # Create the plot with the first y-axis
        fig, ax2 = plt.subplots(figsize = (18, 5))

        # Create the second y - axis for energy storage
        ax1 = ax2.twinx()
        ax2.fill_between(time, en_data, color = 'lavender', label = 'Energy in Battery [MWh]', zorder = 1)

        # Plot power data with distinct colors and line styles for clarity
        ax1.plot(power_peak_data, color = "hotpink", label = "Peak power [MW]", linewidth = 2, zorder=3)
        ax1.plot(power_base_data, color = "royalblue", label = "Base power [MW]", linewidth = 2, zorder=3)
        ax1.plot(power_tot_data, color = "mediumturquoise", label = "Total power [MW]", linewidth = 2, zorder=3)
        ax1.plot(power_to_bat, color = 'lightsteelblue', label = "Power to storage [MW]", linewidth = 2, zorder=3)

        # Set the labels for the first y-axis (Power)
        ax1.set_xlabel("Day of Month", fontsize = 12, labelpad = 10)
        ax1.set_ylabel("Power [MW]", fontsize = 12, labelpad = 10)
        ax1.set_title(f"Power Flow and Stored Energy for Month {i}", fontsize = 14, fontweight = 'bold', pad = 15)

        # Adjust the y-axis limits for better readability
        ax1.set_ylim(0, 600)
        ax1.tick_params(axis = 'y', labelsize = 10)
        ax1.tick_params(axis = 'x', labelsize = 10)

        # Set the range and labels for the second y-axis
        ax2.set_ylim(0, 7000)
        ax2.set_ylabel("Stored Energy [MWh]", fontsize = 12, labelpad = 10)
        ax2.tick_params(axis = 'y', labelsize = 10)

        # Improve the appearance of legends
        ax1.legend(loc = 'upper center', bbox_to_anchor = (0.4, -0.15), fontsize = 10, frameon = False, ncol = 2)
        ax2.legend(loc = 'upper center', bbox_to_anchor = (0.7, -0.15), fontsize = 10, frameon = False)

        ax2.spines['right'].set_color('slateblue')  # Left y-axis spine color
        ax2.yaxis.label.set_color('slateblue')  # Left y-axis label color
        ax2.tick_params(axis = 'y', colors = 'slateblue')  # Left y-axis tick colors

        ax1.spines['left'].set_color('black')  # Right y-axis spine color
        ax1.yaxis.label.set_color('black')  # Right y-axis label color
        ax1.tick_params(axis = 'y', colors = 'black')  # Right y-axis tick colors

        # Adjust the layout to avoid overlap
        plt.tight_layout()

        # Show the plot
        plt.show()


def plot_everything_hour(results, start = 0, stop = 24):
    # Extract data from the results
    hours = list(results['time'].values())[start:stop]
    power_peak = list(results['power peak'].values()).copy()[start:stop]
    power_base = list(results['power base'].values()).copy()[start:stop]
    tot = list(results['power total'].values()).copy()[start:stop]
    battery_energy = list(results['battery'].values()).copy()[start:stop]
    massflow_in = list(results['mass flow in'].values()).copy()[start:stop]
    pow_to_bat = [i * h_delta_turbine / 1000 for i in massflow_in]

    # Create the plot with the first y-axis
    fig, ax2 = plt.subplots(figsize = (18, 5))
    ax1 = ax2.twinx()

    # Plot power data with distinct colors and line styles for clarity
    ax1.plot(hours, power_peak, color = "hotpink", label = "Peak power [MW]", linewidth = 2)
    ax1.plot(hours, power_base, color = "mediumturquoise", label = "Base power [MW]", linewidth = 2)
    ax1.plot(hours, tot, color = "royalblue", label = "Total power [MW]", linewidth = 2)
    ax1.plot(hours, pow_to_bat, color = 'lightsteelblue', label = "Power to storage [MW]", linewidth = 2)

    # Set the labels for the first y-axis (Power)
    ax1.set_xlabel("Number of hour", fontsize = 12, labelpad = 10)
    ax1.set_ylabel("Power [MW]", fontsize = 12, labelpad = 10)
    ax1.set_title(f"Power Flow and Stored Energy for {start}h - {stop}h", fontsize = 14, fontweight = 'bold', pad = 15)

    # Adjust the y-axis limits for better readability
    ax1.set_ylim(0, 800)
    ax1.tick_params(axis = 'y', labelsize = 10)
    ax1.tick_params(axis = 'x', labelsize = 10)

    # Create the second y-axis for power prices (energy storage)
    ax2.fill_between(hours, battery_energy, color = 'lavender', label = 'Energy in Battery [MWh]')

    # Set the range and labels for the second y-axis
    ax2.set_ylim(0, 8500)
    ax2.set_ylabel("Stored Energy [MWh]", fontsize = 12, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 10)

    # Improve the appearance of legends
    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.4, -0.15), fontsize = 10, frameon = False, ncol=2)
    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.7, -0.15), fontsize = 10, frameon = False)

    ax2.spines['right'].set_color('slateblue')  # Right y-axis spine color
    ax2.yaxis.label.set_color('slateblue')  # Right y-axis label color
    ax2.tick_params(axis = 'y', colors = 'slateblue')  # Right y-axis tick colors

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def heat_flows_and_temp(results, specific_heat, start, stop):
    hours = list(results['time'].values())[start:stop]
    q_in = list(results['heat in'].values()).copy()[start:stop]
    q_out = list(results['heat out'].values()).copy()[start:stop]
    q_loss = list(results['loss'].values()).copy()[start:stop]
    battery_energy = list(results['battery'].values()).copy()[start:stop]  # Energy in MWh

    df = pd.DataFrame({'Q in: ': [round(i, 1) for i in q_in],
                       'Q out: ': [round(i, 1) for i in q_out],
                       'Q loss: ': [round(i, 1) for i in q_loss],
                       'Temp (C)': [round(i * 1000 / ((specific_heat / 3600) * 63300000), 1) for i in battery_energy]})

    # Create the plot with the first y-axis
    fig, ax1 = plt.subplots(figsize = (18, 5))

    # Plot power data with distinct colors and line styles for clarity
    ax1.plot(hours, df['Q loss: '], color = "orange", label = "Heat loss [MW]", linewidth = 2)
    ax1.plot(hours, df['Q in: '], color = "forestgreen", label = "Heat inflow [MW]", linewidth = 2)
    ax1.plot(hours, df['Q out: '], color = "royalblue", label = "Heat outflow [MW]", linewidth = 2)

    # Set the labels for the first y-axis (Power)
    ax1.set_xlabel("Hour", fontsize = 12, labelpad = 10)
    ax1.set_ylabel("Heat [MW]", fontsize = 12, labelpad = 10)
    ax1.set_title(f"Heat Flow and Temperature for {start}h - {stop}h", fontsize = 14, fontweight = 'bold', pad = 15)

    # Adjust the y-axis limits for better readability
    ax1.set_ylim(0, 800)
    ax1.tick_params(axis = 'y', labelsize = 10)
    ax1.tick_params(axis = 'x', labelsize = 10)

    # Create the second y-axis for power prices (energy storage)
    ax2 = ax1.twinx()
    ax2.fill_between(hours, df['Temp (C)'], color = 'indianred', label = 'Temperature in Battery [K]', alpha = 0.2)

    # Set the range and labels for the second y-axis
    ax2.set_ylim(0, 600)
    ax2.set_ylabel("Temperature [K]", fontsize = 12, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 10)

    # Improve the appearance of legends
    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.4, -0.15), fontsize = 10, frameon = False, ncol=2)
    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.7, -0.15), fontsize = 10, frameon = False)

    ax2.spines['right'].set_color('cornflowerblue')  # Right y-axis spine color
    ax2.yaxis.label.set_color('cornflowerblue')  # Right y-axis label color
    ax2.tick_params(axis = 'y', colors = 'cornflowerblue')  # Right y-axis tick colors

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def duration(results):
    # Extract data from the results
    hours = list(results['time'].values())
    power_peak = list(results['power peak'].values()).copy()
    power_base = list(results['power base'].values()).copy()
    tot = list(results['power total'].values()).copy()
    battery_energy = list(results['battery'].values()).copy()
    massflow_in = list(results['mass flow in'].values()).copy()
    pow_to_bat = [i * h_delta_turbine / 1000 for i in massflow_in]

    power_peak.sort(reverse = True)
    power_base.sort(reverse = True)
    tot.sort(reverse = True)
    battery_energy.sort(reverse = True)
    massflow_in.sort(reverse = True)
    pow_to_bat.sort(reverse = True)

    # Create the plot with the first y-axis
    fig, ax2 = plt.subplots(figsize = (18, 5))
    ax1 = ax2.twinx()

    # Plot power data with distinct colors and line styles for clarity
    ax1.plot(hours, power_peak, color = "hotpink", label = "Peak power [MW]", linewidth = 2)
    ax1.plot(hours, power_base, color = "mediumturquoise", label = "Base power [MW]", linewidth = 2)
    ax1.plot(hours, tot, color = "royalblue", label = "Total power [MW]", linewidth = 2)
    ax1.plot(hours, pow_to_bat, color = 'lightsteelblue', label = "Power to storage [MW]", linewidth = 2)

    # Set the labels for the first y-axis (Power)
    ax1.set_xlabel("Number of hour", fontsize = 16, labelpad = 10)
    ax1.set_ylabel("Power [MW]", fontsize = 16, labelpad = 10)
    ax1.set_title(f"Sorted Power Dispatch Through the Year", fontsize = 14, fontweight = 'bold', pad = 15)

    # Adjust the y-axis limits for better readability
    ax1.set_ylim(0, 800)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)

    # Create the second y-axis for power prices (energy storage)
    ax2.fill_between(hours, battery_energy, color = 'lavender', label = 'Energy in Battery [MWh]')

    # Set the range and labels for the second y-axis
    ax2.set_ylim(0, 8500)
    ax2.set_ylabel("Stored Energy [MWh]", fontsize = 16, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 10)

    # Improve the appearance of legends
    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.4, -0.3), fontsize = 16, frameon = False, ncol=2)
    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.7, -0.3), fontsize = 16, frameon = False)

    ax2.spines['right'].set_color('slateblue')  # Right y-axis spine color
    ax2.yaxis.label.set_color('slateblue')  # Right y-axis label color
    ax2.tick_params(axis = 'y', colors = 'slateblue')  # Right y-axis tick colors

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_material_bar():
    # Data
    materials = ['Magnesia fire bricks', 'Nitrate salts',
                 'Carbonate salts', 'NaCl (solid)']
    values = [1334.41- 1256.51, 1278.43-1256.51, 1261.73-1256.51, 1256.51-1256.51]

    # Define the colors for each bar (you can change these as needed)
    colors = ['mediumpurple', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue']

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.bar(materials, values, color = colors)
    plt.xlabel('TES Materials')
    plt.ylabel('Profit [MNOK]')
    plt.ylim([0,100])
    #plt.grid(color = "gray", linestyle = "--", linewidth = 0.5, alpha = 0.7)

    #plt.title('Annual benefit of integrating TES')

    # Display numbers on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 10, f'{v:.2f}', ha = 'center', va = 'bottom')

    plt.xticks(rotation = 0, ha = "center")
    plt.tight_layout()
    plt.show()

def plot_efficiency():

    efficiencies =    [1, 0.95, 0.9, 0.85, 0.8, 0.75,0.7, 0.65]
    values =          [1334.4143 - 1256.51, 1313.7646 - 1256.51, 1294.88 - 1256.51, 1281.004 - 1256.51, 1271.1002 - 1256.51, 1264.382 - 1256.51, 1258.9398 - 1256.51, 0]

    plt.figure(figsize = (12, 4))
    plt.grid(color = "gray", linestyle = "--", linewidth = 0.5, alpha = 0.7)

    # Plotting
    plt.plot(efficiencies, values, marker = 'o', linestyle = '-', color = 'palevioletred', linewidth=4)
    plt.xlabel('Efficiencies')
    plt.ylabel('Profit [MNOK]')
    #plt.title('Values vs Efficiencies')

    # Display the plot
    plt.tight_layout()
    plt.show()