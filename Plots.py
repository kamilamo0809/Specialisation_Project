import numpy as np
import matplotlib.pyplot as plt

''' ----- Plotting results ----- '''

def get_lists(model, step1 = 0, tf = 8759):
    hours = range(step1, tf + 1)
    hourslist = list(hours)
    massflow_base = list(pyo.value(model.massflow_base[hour]) for hour in hours)
    massflow_out = list(pyo.value(model.massflow_out[hour]) for hour in hours)
    massflow_in = list(pyo.value(model.massflow_in[hour]) for hour in hours)
    power_peak = list(pyo.value(model.power_peak[hour] / 1000) for hour in hours)
    power_base = list(pyo.value(model.power_base[hour] / 1000) for hour in hours)
    battery_energy = list(pyo.value(model.u_TES[hour] / 1000) for hour in hours)
    total_power = [power_peak[i] + power_base[i] for i in hours]

    everything = [hours, hourslist, massflow_base, massflow_out, massflow_in, power_peak, power_base, battery_energy,total_power]

    return everything


def make_monthly_list(yearly_list):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    all_hours_in_year = []
    start_idx = 0
    for month in days_in_month:
        days = []
        all_hours_in_year.append(yearly_list[start_idx:start_idx + month])
        start_idx += 1
    return all_hours_in_year

def plot_power_output(list_of_mont_numbers, lists):
    hours = lists[0]
    power_peak = lists[4]
    power_base = lists[5]

    time = make_monthly_list(hours)
    TES = make_monthly_list(power_peak)
    base = make_monthly_list(power_base)

    for i in list_of_mont_numbers:
        plt.plot(time[i - 1], TES[i - 1], color = "hotpink", label = "power from battery storage")
        plt.plot(time[i - 1], base[i - 1], color = "mediumpurple", label = "base power")
        plt.xlabel("Time [h]")
        plt.ylabel("Power [kW]")
        plt.title(f"Power output for month number {i}")
        plt.legend()
        plt.show()


def plot_power_output_sorted(lists):
    power_peak = lists[4]
    power_base = lists[5]
    hourslist = lists[0]

    power_peak.sort(reverse = True)
    power_base.sort(reverse = True)

    plt.figure(figsize = (10, 6))  # Set figure size

    # Plot the data
    plt.plot(hourslist, power_peak, color = "mediumturquoise", linewidth = 2, linestyle = "--", label = "Peak Power")
    plt.plot(hourslist, power_base, color = "darkorchid", linewidth = 2, linestyle = "--", label = "Base Power")

    # Fill areas with better visual hierarchy
    plt.fill_between(hourslist, power_base, power_peak, color = "plum", alpha = 0.4, label = "Base Energy Output")
    plt.fill_between(hourslist, power_peak, color = "paleturquoise", alpha = 0.3, label = "Peak Energy Output")

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


def plot_total_power_output_sorted(lists):
    total_power = lists[8]
    hourslist = lists[0]

    total_power.sort(reverse = True)
    plt.plot(hourslist, total_power, color = "lightseagreen", label = "Total power outbut [MW]")
    plt.fill_between(hourslist, total_power, color = 'palegreen', alpha = 0.4)
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kW]")
    plt.title(f"Total power output through the year")
    plt.legend()
    plt.show()

def plot_battery_storage(lists):
    hourslist = lists[0]
    battery_energy = lists[7]
    plt.plot(hourslist, battery_energy, color = 'dodgerblue')
    plt.fill_between(hourslist, battery_energy, color = 'skyblue', alpha = 0.4)
    plt.title("Stored energy in TES through the year")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kWh]")
    plt.show()

def plot_battery_storage_sorted(lists):
    hourslist = lists[0]
    battery_energy = lists[7]

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

# TODO: husk at batteriet har en "duration" siden varme ikke kan lagres særlig lenge
