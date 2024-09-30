import pandas as pd
import os

def read_previous_spot(year: int, zone: str) -> dict:

    # Make dictionary which direct the year to the right excel file
    file_dict = {2021: "spotpriser_21.xlsx",
                 2022: "spotpriser_22.xlsx",
                 2023: "spotpriser_23.xlsx",
                 2050: "new.csv"}

    # Extract the correct file from the dictionary
    filename = file_dict[year]

    # check if the file is csv or excel
    _, file_extension = os.path.splitext(filename)

    if file_extension.lower() == '.csv':
        data = pd.read_csv(filename)
        data = data['Price'] * 1000

    elif file_extension.lower() in ['.xls', '.xlsx']:
        # Read excel
        data = pd.read_excel(filename)
        # Extratc numbers from the right price zone
        data = data[zone] * 100 * 1000  # TODO: check øre/kr and MWh/kWh

    # Convert to dictionary
    price_dict = data.to_dict()

    return price_dict

#print(read_previous_spot(2050, "NO5"))