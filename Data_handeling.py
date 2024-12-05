import matplotlib.pyplot as plt
import pandas as pd
import os

def read_previous_spot(year: int, zone: str) -> dict:

    # Make dictionary which direct the year to the right excel file
    file_dict = {2021: "spotpriser_21.xlsx",
                 2022: "spotpriser_22.xlsx",
                 2023: "spotpriser_23.xlsx",
                 2050: "modified_spot_prices.csv"}

    # Extract the correct file from the dictionary
    filename = file_dict[year]

    # check if the file is csv or excel
    _, file_extension = os.path.splitext(filename)

    if file_extension.lower() == '.csv':
        data = pd.read_csv(filename)
        data = data['Price'] #NOK/kWh

    elif file_extension.lower() in ['.xls', '.xlsx']:
        # Read excel
        data = pd.read_excel(filename)
        # Extratc numbers from the right price zone
        data = data[zone]  # NOK/kWh

    # Convert to dictionary
    price_dict = data.to_dict()

    return price_dict

def plot_spot(year = 2050, zone = 'NO1'):
    print(''' ------------------------ Plotting spot prices ------------------------''')

    spot = read_previous_spot(year = year, zone = zone)

    time = spot.keys()
    price = spot.values()

    zero_hours = sum(1 for i in price if i <= 0)
    no_prod = sum(1 for i in price if i < 0.1)
    print('Number of hours with zero or negative prices: ', zero_hours)
    print('Number of hours where the spot price is lower than the marginal cost of production: ', no_prod)


    # First Plot
    plt.figure(figsize=(10, 6))
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.7)
    plt.plot(time, price, color='mediumslateblue', linewidth=2, label='Spot Price')
    plt.xlabel('Time [hour]', fontsize=14)
    plt.ylabel('Spot Price [NOK/kWh]', fontsize=14)
    plt.title('Spot Price Through The Year', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Second Plot
    plt.figure(figsize=(10, 6))
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.7)
    plt.plot(time, sorted(price, reverse=True), color='cornflowerblue', linewidth=3, label='Sorted Spot Price')
    plt.fill_between(time, sorted(price, reverse=True), color='lightsteelblue', alpha=0.6)
    plt.xlabel('Time [hour]', fontsize=14)
    plt.ylabel('Spot Price [NOK/kWh]', fontsize=14)
    plt.title('Spot prices sorted', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()