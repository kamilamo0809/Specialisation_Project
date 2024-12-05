## TET4510 - Specialization Project
Sizing and Optimized Dispatch of a TES Battery Coupled with a Nuclear Power Plant

### Table of Contents
- [About the project](#about-the-project)
- [List of files](#files)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

### About the Project
The project aims to find the optimal size of a thermal energy storage system coupled to a nuclear power reactor based on 
different factors such as power prices, volatility, storage material properties and investment costs. This model helps stakeholders
and investors estimating the profitability of superflexible nuclear power plants.

### Files
- `Data_handeling.py`: Handles data preprocessing.
- `main.py`: The main script to run the optimization.
- `modified_spot_prices.csv`: Preprocessed spot price data for 2040 scenario.
- `Plots.py`: Script for generating plots and visualizations.
- `Results.csv`: Output results from the optimization.
- `scenario_generation`: Folder containing scenario generation scripts.
- `Sizing_of_TES_Optimization.py`: Core optimization script.
- `spotpriser_21.xlsx`: Historical spot price data for 2021.
- `spotpriser_22.xlsx`: Historical spot price data for 2022.
- `spotpriser_23.xlsx`: Historical spot price data for 2023.

### Getting started
1. Download a copy of the repository:
   ```bash
   git clone https://github.com/kamilamo0809/Specialisation_Project.git
2. Install all required packages:
3. ```bash 
   pip install -r requirements.txt
3. Run the main file:
4. ```bash
   python main.py

### Usage
To run the optimization problem:
1. Open `main.py`.
2. Modify the parameters in the "User Interface" section to customize the settings.
3. Run the file to generate results.

**Note:** To replicate the project results as submitted in the thesis, avoid changing any parameters.

### Contact
* Kamilla Aarflot Moen - [Email](mailto:kamilamo@stud.ntnu.no)

* Project Link: [Link to GitHub repository](https://github.com/kamilamo0809/Specialization_Project)

### Acknowledgments
- [Last year's specialization project](https://github.com/Simend18/TET4510_Project): A similar project that served as a reference and inspiration.
