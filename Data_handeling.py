import pandas as pd

def read_previous_spot(year: int, zone: str) -> dict:

    # Make dictionary which direct the year to the right excel file
    file_dict = {2021: "spotpriser_21.xlsx",
                 2022: "spotpriser_22.xlsx",
                 2023: "spotpriser_23.xlsx"}

    # Extract the correct file from the dictionary
    filename = file_dict[year]
    # Read excel
    data = pd.read_excel(filename)
    # Extratc numbers from the right price zone
    data = data[zone] * 100 # TODO: check øre/kr and MWh/kWh

    # Convert to dictionary
    price_dict = data.to_dict()

    return price_dict

#print(read_previous_spot(2023, "NO5")[5])