'''
This file plots battery output variables to compare and analyse results

Made by Kamilla Aarflot Moen
course: TET4510
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sty import fg
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
        fig, ax2 = plt.subplots(figsize = (18, 3))

        # Create the second y - axis for energy storage
        ax1 = ax2.twinx()
        ax2.fill_between(time, en_data, color = '#dae7f7', label = 'Energy in Battery [MWh]', zorder = 1)

        # Plot power data with distinct colors and line styles for clarity
        ax1.plot(power_peak_data, color = "#ff61bd", label = "Peak power [MW]", linewidth = 3, zorder=3)
        ax1.plot(power_base_data, color = "#b152ff", label = "Base power [MW]", linewidth = 3, zorder=3)
        ax1.plot(power_tot_data, color = "#325cc7", label = "Total power [MW]", linewidth = 3, zorder=3)
        ax1.plot(power_to_bat, color = 'lightsteelblue', label = "Power to storage [MW]", linewidth = 3, zorder=3)

        # Set the labels for the first y-axis (Power)
        ax1.set_xlabel("Day of Month", fontsize = 16, labelpad = 10)
        ax1.set_ylabel("Power [MW]", fontsize = 16, labelpad = 10)
        ax1.set_title(f"Power Flow and Stored Energy for Month {i}", fontsize = 14, fontweight = 'bold', pad = 15)

        # Adjust the y-axis limits for better readability
        ax1.set_ylim(0, 600)
        ax1.tick_params(axis = 'y', labelsize = 14)
        ax1.tick_params(axis = 'x', labelsize = 14)

        # Set the range and labels for the second y-axis
        ax2.set_ylim(0, 7000)
        ax2.set_ylabel("Stored Energy [MWh]", fontsize = 16, labelpad = 10)
        ax2.tick_params(axis = 'y', labelsize = 14)

        # Improve the appearance of legends
        #ax1.legend(loc = 'upper center', bbox_to_anchor = (0.4, -0.15), fontsize = 10, frameon = False, ncol = 2)
        #ax2.legend(loc = 'upper center', bbox_to_anchor = (0.7, -0.15), fontsize = 10, frameon = False)

        ax2.spines['right'].set_color('#87a5cc')  # Left y-axis spine color
        ax2.yaxis.label.set_color('#87a5cc')  # Left y-axis label color
        ax2.tick_params(axis = 'y', colors = '#87a5cc')  # Left y-axis tick colors

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
    fig, ax2 = plt.subplots(figsize = (18, 3))
    ax1 = ax2.twinx()

    # Plot power data with distinct colors and line styles for clarity
    ax1.plot(hours, power_peak, color = "#ff61bd", label = "Peak power [MW]", linewidth = 3)
    ax1.plot(hours, power_base, color = "#b152ff", label = "Base power [MW]", linewidth = 3)
    ax1.plot(hours, tot, color = "#325cc7", label = "Total power [MW]", linewidth = 3)
    ax1.plot(hours, pow_to_bat, color = 'lightsteelblue', label = "Power to storage [MW]", linewidth = 3)

    # Set the labels for the first y-axis (Power)
    ax1.set_xlabel("Number of hour", fontsize = 16, labelpad = 10)
    ax1.set_ylabel("Power [MW]", fontsize = 16, labelpad = 10)
    ax1.set_title(f"Power Flow and Stored Energy for {start}h - {stop}h", fontsize = 14, fontweight = 'bold', pad = 15)

    # Adjust the y-axis limits for better readability
    ax1.set_ylim(0, 800)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)

    # Create the second y-axis for power prices (energy storage)
    ax2.fill_between(hours, battery_energy, color = '#dae7f7', label = 'Energy in Battery [MWh]')

    # Set the range and labels for the second y-axis
    ax2.set_ylim(0, 8500)
    ax2.set_ylabel("Stored Energy [MWh]", fontsize = 16, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 14)

    # Improve the appearance of legends
    #ax1.legend(loc = 'upper center', bbox_to_anchor = (0.4, -0.15), fontsize = 10, frameon = False, ncol=2)
    #ax2.legend(loc = 'upper center', bbox_to_anchor = (0.7, -0.15), fontsize = 10, frameon = False)

    ax2.spines['right'].set_color('#87a5cc')  # Right y-axis spine color
    ax2.yaxis.label.set_color('#87a5cc')  # Right y-axis label color
    ax2.tick_params(axis = 'y', colors = '#87a5cc')  # Right y-axis tick colors

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
    ax1.plot(hours, power_peak, color = "#ff61bd", label = "Peak power [MW]", linewidth = 3)
    ax1.plot(hours, power_base, color = "#b152ff", label = "Base power [MW]", linewidth = 3)
    ax1.plot(hours, tot, color = "#325cc7", label = "Total power [MW]", linewidth = 3)
    ax1.plot(hours, pow_to_bat, color = 'lightsteelblue', label = "Power to storage [MW]", linewidth = 3)

    # Set the labels for the first y-axis (Power)
    ax1.set_xlabel("Number of hour", fontsize = 16, labelpad = 10)
    ax1.set_ylabel("Power [MW]", fontsize = 16, labelpad = 10)
    ax1.set_title(f"Sorted Power Dispatch Through the Year", fontsize = 14, fontweight = 'bold', pad = 15)

    # Adjust the y-axis limits for better readability
    ax1.set_ylim(0, 800)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)

    # Create the second y-axis for power prices (energy storage)
    ax2.fill_between(hours, battery_energy, color = '#dae7f7', label = 'Energy in Battery [MWh]')

    # Set the range and labels for the second y-axis
    ax2.set_ylim(0, 8500)
    ax2.set_ylabel("Stored Energy [MWh]", fontsize = 16, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 14)

    # Improve the appearance of legends
    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.4, -0.3), fontsize = 16, frameon = False, ncol=2)
    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.7, -0.3), fontsize = 16, frameon = False)

    ax2.spines['right'].set_color('#87a5cc')  # Right y-axis spine color
    ax2.yaxis.label.set_color('#87a5cc')  # Right y-axis label color
    ax2.tick_params(axis = 'y', colors = '#87a5cc')  # Right y-axis tick colors

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_material_bar():
    # Data
    materials = ['Magnesia fire bricks', 'Nitrate salts',
                 'Carbonate salts', 'NaCl (solid)']
    values = [1334.41- 1256.51, 1278.43-1256.51, 1261.73-1256.51, 1256.51-1256.51]

    peak = [(300 + 126.34)/300, 1.419, 1.419, 1]
    pros_peak = [i * 100 for i in peak]

    fig, ax1 = plt.subplots(figsize = (16, 4))
    ax2 = ax1.twinx()

    # Define the positions and bar width
    x = np.arange(len(materials))
    bar_width = 0.45

    # Bars for profit values
    ax1.bar(x - bar_width / 2, values, width = bar_width, color = '#e37898', label = 'Net benefit')
    ax1.set_ylabel('Net benefit [MNOK]', fontsize = 16, color = '#bd4065')
    ax1.tick_params(axis = 'y', labelcolor = '#bd4065', labelsize = 14)
    ax1.spines['right'].set_color('#bd4065')
    ax1.set_ylim([0, 90])

    # Secondary axis for peak capacity
    ax2.bar(x + bar_width / 2, pros_peak, width = bar_width, color = '#ffc894', label = 'Peak Output Capacity [%]')
    ax2.set_ylabel('Peak Output Capacity [%]', fontsize = 16, color = '#e69143')
    ax2.tick_params(axis = 'y', labelcolor = '#e69143', labelsize = 14)
    ax2.spines['right'].set_color('#e69143')

    # X-axis labels
    ax1.set_xticks(x)
    ax2.set_ylim([99, 160])
    ax1.set_xticklabels(materials, fontsize = 14)
    ax1.set_xlabel('TES Materials', fontsize = 16)

    # Display numbers on top of each bar
    for i, (v1, v2) in enumerate(zip(values, pros_peak)):
        ax1.text(i - bar_width / 2, v1 + 2, f'{v1:.2f} MNOK', ha = 'center', fontsize = 10)
        ax2.text(i + bar_width / 2, v2 + 2, f'{v2:.2f} %', ha = 'center', fontsize = 10)

    # Add legend
    fig.legend(loc = 'upper right', bbox_to_anchor = (0.95, 0.95), bbox_transform = ax1.transAxes)

    plt.tight_layout()

    plt.show()


def plot_efficiency():

    efficiencies =    [1, 0.95 * 0.95, 0.9 * 0.9, 0.85 * 0.85, 0.8 * 0.8, 0.75 * 0.75, 0.7 * 0.7, 0.65 * 0.65]
    efficiencies = [i * 100 for i in efficiencies]
    values =          [1334.4143 - 1256.51, 1313.7646 - 1256.51, 1294.88 - 1256.51, 1281.004 - 1256.51, 1271.1002 - 1256.51, 1264.382 - 1256.51, 1258.9398 - 1256.51, 0]
    peak = [(300 + 126.34)/300, (300 + 97.49)/300 ,(300 + 72.48)/300, (300 + 55.63)/300, (300 + 43.19)/300, (300 + 35.86)/300, (300 + 29.96)/300, (300 + 0)/300]
    pros_peak = [i * 100 for i in peak]

    fig, ax1 = plt.subplots(figsize = (16, 4))
    ax2 = ax1.twinx()

    # Plotting
    x = np.arange(len(efficiencies))
    bar_width = 0.4

    # First bar group (Profit)
    ax1.bar(x - bar_width / 2, values, width = bar_width, color = '#e37898', linewidth = 4,
            label = "Profit [MNOK]")

    # Second bar group (Peak output capacity)
    ax2.bar(x + bar_width / 2, pros_peak, width = bar_width, color = '#ffc894', label = "Peak Output Capacity [%]")

    # Set the labels for the first y-axis (MNOK)
    ax1.set_xlabel("Round-trip efficiency [%]", fontsize = 16, labelpad = 10)
    ax1.set_ylabel("Profit [MNOK]", fontsize = 16, labelpad = 10)

    # Set the x-axis tick positions and labels
    plt.xticks(x, [round(i, 1) for i in efficiencies])

    # Adjust the y-axis limits for better readability
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.set_ylim([0,90])

    # Set the range and labels for the second y-axis
    ax2.set_ylabel("Peak output capacity [%]", fontsize = 16, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 10)

    # Style adjustments for clarity
    ax2.spines['right'].set_color('#e69143')  # Right y-axis spine color
    ax2.yaxis.label.set_color('#e69143')  # Right y-axis label color
    ax2.tick_params(axis = 'y', colors = '#e69143')  # Right y-axis tick colors
    ax2.set_ylim([95, 155])

    ax1.spines['left'].set_color('#bd4065')  # Left y-axis spine color
    ax1.yaxis.label.set_color('#bd4065')  # Left y-axis label color
    ax1.tick_params(axis = 'y', colors = '#bd4065')  # Left y-axis tick colors

    # Display numbers on top of each bar
    for i, (v1, v2) in enumerate(zip(values, pros_peak)):
        ax1.text(i - bar_width / 2, v1 + 2, f'{v1:.2f}', ha = 'center', fontsize = 10)
        ax2.text(i + bar_width / 2, v2 + 2, f'{v2:.1f}%', ha = 'center', fontsize = 10)

    # Add legend
    fig.legend(loc = 'upper right', bbox_to_anchor = (0.95, 0.95), bbox_transform = ax1.transAxes)

    # Display the plot
    plt.tight_layout()

    plt.show()


def plot_everything_month_aggregated(results, year, zone):
    # Extract data from the results
    hours = list(results['time'].values())
    power_peak = list(results['power peak'].values())
    power_base = list(results['power base'].values())
    battery_energy = list(results['battery'].values())
    power_prices = read_previous_spot(year, zone)
    power_prices = [power_prices[i] for i in power_prices]

    # Create monthly list from the yearly data
    power_peak_monthly = make_monthly_list(power_peak)
    power_base_monthly = make_monthly_list(power_base)
    battery_energy_monthly = make_monthly_list(battery_energy)
    power_prices_monthly = make_monthly_list(power_prices)

    # Aggregated montly values in lists
    peak_aggregated = [sum(power_peak_monthly[i])/1000 for i in range(12)]
    base_aggregated = [sum(power_base_monthly[i])/1000 for i in range(12)]
    batt_aggregated = [sum(battery_energy_monthly[i])/1000 for i in range(12)]
    price_average = [sum(power_prices_monthly[i])/len(power_prices_monthly[i]) for i in range(12)]

    # Create a figure with two subplots side by side
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize = (16, 8))  # 1 row, 2 columns

    # ----------------------------- FIRST PLOT ------------------------

    # First y-axis for energy dispatch
    ax2 = ax1.twinx()

    # Bar plots for base and peak aggregated power
    ax1.bar(list(range(1, 13)), base_aggregated, color = "xkcd:rosa", label = "Aggregated base power")
    ax1.bar(list(range(1, 13)), peak_aggregated, color = "xkcd:rosy pink", label = "Aggregated peak power",
            bottom = base_aggregated)

    # Line plot for average electricity price
    ax2.plot(list(range(1, 13)), price_average, color = 'xkcd:ocean', linewidth = 4,
             label = 'Average electricity price')

    # Set labels and styling for first plot
    ax1.set_xlabel("Month", fontsize = 14, labelpad = 10)
    ax1.set_ylabel("Energy [GWh]", fontsize = 14, labelpad = 10)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)

    ax2.set_ylim(0, 1.7)
    ax2.tick_params(axis = 'y', labelsize = 14)

    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.25, -0.15), fontsize = 12, frameon = False)
    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.8, -0.15), fontsize = 12, frameon = False)

    # ----------------------------- SECOND PLOT ------------------------

    # Second y-axis for energy storage
    ax4 = ax3.twinx()

    # Bar plot for stored energy
    ax3.bar(list(range(1, 13)), batt_aggregated, color = "xkcd:light grey green",
            label = "Aggregated stored energy [GWh]")

    # Line plot for average electricity price
    ax4.plot(list(range(1, 13)), price_average, color = 'xkcd:ocean', linewidth = 4,
             label = 'Average electricity price')

    # Set labels and styling for second plot
    ax3.set_xlabel("Month", fontsize = 14, labelpad = 10)
    ax3.tick_params(axis = 'y', labelsize = 14)
    ax3.tick_params(axis = 'x', labelsize = 14)

    ax4.set_ylim(0, 1.7)
    ax4.set_ylabel("Cost of power [NOK/kWh]", fontsize = 14, labelpad = 10)
    ax4.tick_params(axis = 'y', labelsize = 14)

    ax3.legend(loc = 'upper center', bbox_to_anchor = (0.25, -0.15), fontsize = 12, frameon = False)
    ax4.legend(loc = 'upper center', bbox_to_anchor = (0.8, -0.15), fontsize = 12, frameon = False)

    # ----------------------------- Adjust and Display ------------------------

    # Add spacing between the plots
    plt.tight_layout(w_pad = 5)

    # Show the combined figure
    plt.show()


def plot_CF_and_VF(year, zone):
    prod_with_TES = list(pd.read_csv(f'res{year}True.csv').to_dict(orient='dict')['power total'].values()).copy()
    prod_no_TES = list(pd.read_csv(f'res{year}False.csv').to_dict(orient='dict')['power total'].values()).copy()
    prod_cap = [300 for i in range(8760)]
    power_prices = read_previous_spot(year, zone)
    power_prices = [i for i in power_prices.values()]

    # Create monthly list from the yearly data
    production_monthly_with_TES = make_monthly_list(prod_with_TES)
    production_monthly_no_TES = make_monthly_list(prod_no_TES)
    prod_cap_monthly = make_monthly_list(prod_cap)
    monthly_el_price = make_monthly_list(power_prices)

    #################### CALCULATE CAPACITY FACTOR ###############################

    # Annual production / (8760h * production capacity)

    # Calculate monthly means
    prod_mean_with_TES = [sum(production_monthly_with_TES[i]) for i in range(12)]
    prod_mean_no_TES = [sum(production_monthly_no_TES[i]) for i in range(12)]
    production_capacity_mean = [sum(prod_cap_monthly[i]) for i in range(12)]

    # Calculate
    CF_with_TES = [prod_mean_with_TES[i]/production_capacity_mean[i] * 100 for i in range(12)]
    CF_no_TES = [prod_mean_no_TES[i]/production_capacity_mean[i] * 100 for i in range(12)]

    # Print yearly results
    print(fg(215) + f"Capacity factor with TES: {round(sum(prod_with_TES[i] for i in range(8760)) / (8760 * 300) * 100, 2)} %" + fg.rs)
    print(fg(215) + f"Capacity factor without TES: {round(sum(prod_no_TES[i] for i in range(8760)) / (8760 * 300) * 100, 2)} %" + fg.rs)

    #################### CALCULATE VALUE FACTOR ###############################

    # Remove division by 0
    production_monthly_with_TES = [[0.001 if value == 0 else value for value in sublist] for sublist in production_monthly_with_TES]
    production_monthly_no_TES = [[0.001 if value == 0 else value for value in sublist] for sublist in production_monthly_no_TES]

    time = list(range(12))

    VF_with_TES = []
    VF_no_TES = []
    for i in range(12):
        Average_achieved_el_price_with_TES = sum(monthly_el_price[i][k] * production_monthly_with_TES[i][k] for k in range(len(monthly_el_price[i]))) / (sum(production_monthly_with_TES[i]))
        Average_achieved_el_price_no_TES = sum(monthly_el_price[i][k] * production_monthly_no_TES[i][k] for k in range(len(monthly_el_price[i]))) / (sum(production_monthly_no_TES[i]))

        Average_el_price = sum(monthly_el_price[i]) / len(monthly_el_price[i])
        VF_with_TES.append(Average_achieved_el_price_with_TES / Average_el_price * 100)
        VF_no_TES.append(Average_achieved_el_price_no_TES / Average_el_price * 100)

    # Calculate and print VF for entire year
    Average_achieved_el_price_with_TES = sum(power_prices[i] * prod_with_TES[i] for i in range(8760)) / (sum(prod_with_TES))
    Average_achieved_el_price_no_TES = sum(power_prices[i] * prod_no_TES[i] for i in range(8760)) / (sum(prod_no_TES))

    Average_el_price = sum(power_prices)/8760
    VF_with_TES_year = Average_achieved_el_price_with_TES / Average_el_price
    VF_no_TES_year = Average_achieved_el_price_no_TES / Average_el_price

    print(fg(217) + f"Value factor with TES: {round(VF_with_TES_year * 100, 2)} %" + fg.rs)
    print(fg(217) + f"Value factor without TES: {round(VF_no_TES_year * 100, 2)} %" + fg.rs)


    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))  # 1 row, 2 columns

    # ----------------------------- FIRST PLOT ------------------------

    # Bar plots for base and peak aggregated power
    ax1.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)
    ax1.plot(time, CF_no_TES, color='xkcd:sky', linewidth = 4, marker = 'o', linestyle = 'dashed', label = 'No storage')
    ax1.plot(time, CF_with_TES, color='xkcd:ocean', linewidth = 4, marker = 'o', label = 'Nitrate salts storage')

    # Set labels and styling for first plot
    ax1.set_xlabel("Month", fontsize = 16, labelpad = 10)
    ax1.set_ylabel("Capacity factor [%]", fontsize = 16, labelpad = 10)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)

    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False, ncols = 2)

    # ----------------------------- SECOND PLOT ------------------------

    # Bar plots for base and peak aggregated power
    ax2.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)
    ax2.plot(time, VF_no_TES, color='xkcd:sky', linewidth = 4, marker = 'o', linestyle = 'dashed', label = 'No storage')
    ax2.plot(time, VF_with_TES, color='xkcd:ocean', linewidth = 4, marker = 'o', label = 'Nitrate salts')

    # Set labels and styling for first plot
    ax2.set_xlabel("Month", fontsize = 16, labelpad = 10)
    ax2.set_ylabel("Value factor [%]", fontsize = 16, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax2.tick_params(axis = 'x', labelsize = 14)

    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False, ncols = 2)

    # ----------------------------- Adjust and Display ------------------------

    # Add spacing between the plots
    plt.tight_layout(w_pad = 5)

    # Show the combined figure
    plt.show()

    return VF_with_TES, VF_no_TES, CF_with_TES, CF_no_TES


def plot_diff_VF_CF():
    VF_with_TES_s1, VF_no_TES_s1, CF_with_TES_s1, CF_no_TES_s1 = plot_CF_and_VF(2023, 'NO1')
    VF_with_TES_s2, VF_no_TES_s2, CF_with_TES_s2, CF_no_TES_s2 = plot_CF_and_VF(2040, 'NO1')

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))  # 1 row, 2 columns
    time = list(range(12))

    # ----------------------------- FIRST PLOT ------------------------

    # Bar plots for base and peak aggregated power
    ax1.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)

    width = 0.4
    # Plot the bars beside each other by adjusting their positions
    ax1.bar([i - width / 2 for i in time], [CF_with_TES_s1[i] - CF_no_TES_s1[i] for i in range(12)], width = width,
            color = 'xkcd:light grey green', label = 'Difference in CF in scenario 1')
    ax1.bar([i + width / 2 for i in time], [CF_with_TES_s2[i] - CF_no_TES_s2[i] for i in range(12)], width = width,
            color = 'xkcd:pink', label = 'Difference in CF in scenario 2')

    # Adjust x-axis ticks to be in the middle of the grouped bars
    ax1.set_xticks(time)
    ax1.set_xticklabels([str(i + 1) for i in range(12)])  # Labels for months
    # Set labels and styling for first plot
    ax1.set_xlabel("Month", fontsize = 16, labelpad = 10)
    ax1.set_ylabel("Capacity factor [%]", fontsize = 16, labelpad = 10)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)

    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False)

    # ----------------------------- SECOND PLOT ------------------------

    # Bar plots for base and peak aggregated power
    ax2.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)
    ax2.bar([i - width / 2 for i in time], [VF_with_TES_s1[i] - VF_no_TES_s1[i] for i in range(12)], color='xkcd:light grey green',  width = width, label = 'Difference in VF in scenario 1')
    ax2.bar([i + width / 2 for i in time], [VF_with_TES_s2[i] - VF_no_TES_s2[i] for i in range(12)], color='xkcd:pink',  width = width, label = 'Difference in VF in scenario 2')

    # Set labels and styling for first plot
    ax2.set_xlabel("Month", fontsize = 16, labelpad = 10)
    ax2.set_ylabel("Value factor [%]", fontsize = 16, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax2.tick_params(axis = 'x', labelsize = 14)

    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False)

    # ----------------------------- Adjust and Display ------------------------

    # Add spacing between the plots
    plt.tight_layout(w_pad = 5)

    # Show the combined figure
    plt.show()


def plot_CF():
    prod_with_TES_2023 = list(pd.read_csv(f'res2023True.csv').to_dict(orient='dict')['power total'].values()).copy()
    prod_no_TES_2023 = list(pd.read_csv(f'res2023False.csv').to_dict(orient='dict')['power total'].values()).copy()
    prod_with_TES_2040 = list(pd.read_csv(f'res2040True.csv').to_dict(orient='dict')['power total'].values()).copy()
    prod_no_TES_2040 = list(pd.read_csv(f'res2040False.csv').to_dict(orient='dict')['power total'].values()).copy()
    prod_cap = [300 for i in range(8760)]

    # Create monthly list from the yearly data
    production_monthly_with_TES_2023 = make_monthly_list(prod_with_TES_2023)
    production_monthly_no_TES_2023 = make_monthly_list(prod_no_TES_2023)
    production_monthly_with_TES_2040 = make_monthly_list(prod_with_TES_2040)
    production_monthly_no_TES_2040 = make_monthly_list(prod_no_TES_2040)
    prod_cap_monthly = make_monthly_list(prod_cap)

    #################### CALCULATE CAPACITY FACTOR ###############################

    # Annual production / (8760h * production capacity)

    # Calculate monthly means
    prod_mean_with_TES_2023 = [sum(production_monthly_with_TES_2023[i]) for i in range(12)]
    prod_mean_no_TES_2023 = [sum(production_monthly_no_TES_2023[i]) for i in range(12)]
    prod_mean_with_TES_2040 = [sum(production_monthly_with_TES_2040[i]) for i in range(12)]
    prod_mean_no_TES_2040 = [sum(production_monthly_no_TES_2040[i]) for i in range(12)]
    production_capacity_mean = [sum(prod_cap_monthly[i]) for i in range(12)]

    # Calculate
    CF_with_TES_2023 = [prod_mean_with_TES_2023[i]/production_capacity_mean[i] * 100 for i in range(12)]
    CF_no_TES_2023 = [prod_mean_no_TES_2023[i]/production_capacity_mean[i] * 100 for i in range(12)]
    CF_with_TES_2040 = [prod_mean_with_TES_2040[i] / production_capacity_mean[i] * 100 for i in range(12)]
    CF_no_TES_2040 = [prod_mean_no_TES_2040[i] / production_capacity_mean[i] * 100 for i in range(12)]

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))  # 1 row, 2 columns

    time = list(range(12))

    # ----------------------------- FIRST PLOT ------------------------

    # Bar plots for base and peak aggregated power
    ax1.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)
    ax1.plot(time, CF_no_TES_2023, color='xkcd:sky', linewidth = 4, marker = 'o', linestyle = 'dashed', label = 'w/o storage')
    ax1.plot(time, CF_with_TES_2023, color='xkcd:ocean', linewidth = 4, marker = 'o', label = 'w/ storage')

    # Set labels and styling for first plot
    ax1.set_xlabel("Month in scenario 1", fontsize = 16, labelpad = 10)
    ax1.set_ylabel("Capacity factor [%]", fontsize = 16, labelpad = 10)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)

    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False, ncols = 2)

    # ----------------------------- SECOND PLOT ------------------------

    # Bar plots for base and peak aggregated power
    ax2.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)
    ax2.plot(time, CF_no_TES_2040, color='xkcd:sky', linewidth = 4, marker = 'o', linestyle = 'dashed', label = 'w/o storage')
    ax2.plot(time, CF_with_TES_2040, color='xkcd:ocean', linewidth = 4, marker = 'o', label = 'w storage')

    # Set labels and styling for first plot
    ax2.set_xlabel("Month in scenario 2", fontsize = 16, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax2.tick_params(axis = 'x', labelsize = 14)

    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False, ncols = 2)

    # ----------------------------- Adjust and Display ------------------------

    # Add spacing between the plots
    plt.tight_layout(w_pad = 5)

    # Show the combined figure
    plt.show()


def plot_VF():
    prod_with_TES_2023 = list(pd.read_csv(f'res{2023}True.csv').to_dict(orient='dict')['power total'].values()).copy()
    prod_no_TES_2023 = list(pd.read_csv(f'res{2023}False.csv').to_dict(orient='dict')['power total'].values()).copy()
    power_prices_2023 = read_previous_spot(2023, 'NO1')
    power_prices_2023 = [i for i in power_prices_2023.values()]

    prod_with_TES_2040 = list(pd.read_csv(f'res{2040}True.csv').to_dict(orient='dict')['power total'].values()).copy()
    prod_no_TES_2040 = list(pd.read_csv(f'res{2040}False.csv').to_dict(orient='dict')['power total'].values()).copy()
    power_prices_2040 = read_previous_spot(2040, 'NO1')
    power_prices_2040 = [i for i in power_prices_2040.values()]

    # Create monthly list from the yearly data
    production_monthly_with_TES_2023 = make_monthly_list(prod_with_TES_2023)
    production_monthly_no_TES_2023 = make_monthly_list(prod_no_TES_2023)
    monthly_el_price_2023 = make_monthly_list(power_prices_2023)

    production_monthly_with_TES_2040 = make_monthly_list(prod_with_TES_2040)
    production_monthly_no_TES_2040 = make_monthly_list(prod_no_TES_2040)
    monthly_el_price_2040 = make_monthly_list(power_prices_2040)

    #################### CALCULATE VALUE FACTOR ###############################

    # Remove division by 0
    production_monthly_with_TES_2023 = [[0.001 if value == 0 else value for value in sublist] for sublist in production_monthly_with_TES_2023]
    production_monthly_no_TES_2023 = [[0.001 if value == 0 else value for value in sublist] for sublist in production_monthly_no_TES_2023]
    production_monthly_with_TES_2040 = [[0.001 if value == 0 else value for value in sublist] for sublist in production_monthly_with_TES_2040]
    production_monthly_no_TES_2040 = [[0.001 if value == 0 else value for value in sublist] for sublist in production_monthly_no_TES_2040]
    time = list(range(12))

    VF_with_TES_2023 = []
    VF_no_TES_2023 = []
    for i in range(12):
        Average_achieved_el_price_with_TES_2023 = sum(monthly_el_price_2023[i][k] * production_monthly_with_TES_2023[i][k] for k in range(len(monthly_el_price_2023[i]))) / (sum(production_monthly_with_TES_2023[i]))
        Average_achieved_el_price_no_TES_2023 = sum(monthly_el_price_2023[i][k] * production_monthly_no_TES_2023[i][k] for k in range(len(monthly_el_price_2023[i]))) / (sum(production_monthly_no_TES_2023[i]))

        Average_el_price_2023 = sum(monthly_el_price_2023[i]) / len(monthly_el_price_2023[i])
        VF_with_TES_2023.append(Average_achieved_el_price_with_TES_2023 / Average_el_price_2023 * 100)
        VF_no_TES_2023.append(Average_achieved_el_price_no_TES_2023 / Average_el_price_2023 * 100)


    VF_with_TES_2040 = []
    VF_no_TES_2040 = []
    for i in range(12):
        Average_achieved_el_price_with_TES_2040 = sum(
            monthly_el_price_2040[i][k] * production_monthly_with_TES_2040[i][k] for k in range(len(monthly_el_price_2040[i]))) / (
                                                 sum(production_monthly_with_TES_2040[i]))
        Average_achieved_el_price_no_TES_2040 = sum(
            monthly_el_price_2040[i][k] * production_monthly_no_TES_2040[i][k] for k in range(len(monthly_el_price_2040[i]))) / (
                                               sum(production_monthly_no_TES_2040[i]))

        Average_el_price_2040 = sum(monthly_el_price_2040[i]) / len(monthly_el_price_2040[i])
        VF_with_TES_2040.append(Average_achieved_el_price_with_TES_2040 / Average_el_price_2040 * 100)
        VF_no_TES_2040.append(Average_achieved_el_price_no_TES_2040 / Average_el_price_2040 * 100)


    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))  # 1 row, 2 columns

    # ----------------------------- FIRST PLOT ------------------------

    # Bar plots for base and peak aggregated power
    ax1.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)
    ax1.plot(time, VF_no_TES_2023, color='xkcd:sky', linewidth = 4, marker = 'o', linestyle = 'dashed', label = 'w/o storage')
    ax1.plot(time, VF_with_TES_2023, color='xkcd:ocean', linewidth = 4, marker = 'o', label = 'w/ storage')

    # Set labels and styling for first plot
    ax1.set_xlabel("Month in scenario 1", fontsize = 16, labelpad = 10)
    ax1.set_ylabel("Value factor [%]", fontsize = 16, labelpad = 10)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.tick_params(axis = 'x', labelsize = 14)

    ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False, ncols = 2)

    # ----------------------------- SECOND PLOT ------------------------

    # Bar plots for base and peak aggregated power
    ax2.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)
    ax2.plot(time, VF_no_TES_2040, color='xkcd:sky', linewidth = 4, marker = 'o', linestyle = 'dashed', label = 'w/o storage')
    ax2.plot(time, VF_with_TES_2040, color='xkcd:ocean', linewidth = 4, marker = 'o', label = 'w/ storage')

    # Set labels and styling for first plot
    ax2.set_xlabel("Month in scenario 2", fontsize = 16, labelpad = 10)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax2.tick_params(axis = 'x', labelsize = 14)

    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False, ncols = 2)

    # ----------------------------- Adjust and Display ------------------------

    # Add spacing between the plots
    plt.tight_layout(w_pad = 5)

    # Show the combined figure
    plt.show()


def plot_diff_VF_and_CF():
    VF_with_TES_s1, VF_no_TES_s1, CF_with_TES_s1, CF_no_TES_s1 = plot_CF_and_VF(2023, 'NO1')
    VF_with_TES_s2, VF_no_TES_s2, CF_with_TES_s2, CF_no_TES_s2 = plot_CF_and_VF(2040, 'NO1')

    # Create a figure with two subplots side by side
    time = list(range(12))

    # ----------------------------- FIRST PLOT ------------------------
    plt.figure(figsize = (16, 8))

    # Bar plots for base and peak aggregated power
    plt.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)

    width = 0.4
    # Plot the bars beside each other by adjusting their positions
    plt.bar([i - width / 2 for i in time], [CF_with_TES_s1[i] - CF_no_TES_s1[i] for i in range(12)], width = width,
            color = 'xkcd:sky', label = 'Difference in CF in scenario 1')
    plt.bar([i + width / 2 for i in time], [CF_with_TES_s2[i] - CF_no_TES_s2[i] for i in range(12)], width = width,
            color = 'xkcd:ocean', label = 'Difference in CF in scenario 2')

    # Adjust x-axis ticks to be in the middle of the grouped bars
    #plt.tick_params(time)
    #plt.xticks([str(i + 1) for i in range(12)])  # Labels for months
    # Set labels and styling for first plot
    plt.xlabel("Month", fontsize = 16, labelpad = 10)
    plt.ylabel("Capacity factor [%]", fontsize = 16, labelpad = 10)
    plt.tick_params(axis = 'y', labelsize = 14)
    plt.tick_params(axis = 'x', labelsize = 14)

    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False, ncol = 2)

    plt.tight_layout(w_pad = 5)

    # Show the combined figure
    plt.show()
    # ----------------------------- SECOND PLOT ------------------------
    plt.figure(figsize = (16, 8))

    # Bar plots for base and peak aggregated power
    plt.grid(color = "xkcd:light grey blue", linestyle = "--", linewidth = 0.5)
    plt.bar([i - width / 2 for i in time], [VF_with_TES_s1[i] - VF_no_TES_s1[i] for i in range(12)], color='xkcd:sky',  width = width, label = 'Difference in VF in scenario 1')
    plt.bar([i + width / 2 for i in time], [VF_with_TES_s2[i] - VF_no_TES_s2[i] for i in range(12)], color='xkcd:ocean',  width = width, label = 'Difference in VF in scenario 2')

    # Set labels and styling for first plot
    plt.xlabel("Month", fontsize = 16, labelpad = 10)
    plt.ylabel("Value factor [%]", fontsize = 16, labelpad = 10)
    #plt.tick_params(axis = 'y', labelsize = 14)
    #plt.tick_params(axis = 'x', labelsize = 14)

    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fontsize = 16, frameon = False, ncol = 2)

    # ----------------------------- Adjust and Display ------------------------

    # Add spacing between the plots
    plt.tight_layout(w_pad = 5)

    # Show the combined figure
    plt.show()
