import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)  # Replace 42 with any integer seed of your choice

# Load your 2023 spot prices from CSV (make sure the 'price' column is present)
data = pd.read_excel("spotpriser_23.xlsx")
prices_2023 = data['NO1'].values

# Step 1: Define parameters
zero_negative_ratio = 0.115  # 30% zero or negative prices
target_mean = 0.6  # Desired mean price

# Step 2: Count how many zero or negative prices are already in the 2023 data
num_zero_negative_existing = np.sum(prices_2023 <= 0)

# Step 3: Calculate how many more zero or negative prices we need
num_prices = len(prices_2023)
num_zero_negative_needed = int(zero_negative_ratio * num_prices)
num_zero_negative_to_add = num_zero_negative_needed - num_zero_negative_existing

# Step 4: Create the zero or negative prices
# Randomly select the indices to be set to zero or negative values
if num_zero_negative_to_add > 0:
    zero_negative_indices = np.random.choice(np.where(prices_2023 > 0)[0], size=num_zero_negative_to_add, replace=False)
else:
    zero_negative_indices = []

# Step 5: Scale prices around the pattern of 2023 prices
# Normalize the original prices so that their mean is 0 (zero-centered)
mean_2023 = np.mean(prices_2023)
scaled_prices = prices_2023 - mean_2023  # Zero-centered prices

# Step 6: Add a target mean
scaled_prices += target_mean

# Step 7: Set the required number of prices to zero or negative values
modified_prices = scaled_prices.copy()
if num_zero_negative_to_add > 0:
    modified_prices[zero_negative_indices] = np.random.uniform(low=-1.5, high=0, size=num_zero_negative_to_add)

# Step 9: Verify the mean and pattern
print(f"New mean price: {np.mean(modified_prices)}")
print(f"Number of zero or negative prices: {np.sum(modified_prices <= 0)}, precantage: {round(np.sum(modified_prices <= 0)/8760 * 100, 1)}%")
no_prod = sum(1 for i in modified_prices if i < 0.1)
print('Number of hours where the spot price is lower than the marginal cost of production: ', no_prod)

# Step 10: Save the modified prices to a new CSV
# Create a DataFrame with 'Hour' as the index and 'Price' as the price column
df = pd.DataFrame({
    'Hour': range(1, len(modified_prices) + 1),  # Assuming you want hours from 1 to n
    'Price': modified_prices
})
df.to_csv("modified_spot_prices.csv", index=False)


# Step 11: Plot
plt.figure(figsize=(10, 6))
plt.grid(color='lightgrey', linestyle='--', linewidth=0.7)
plt.plot(modified_prices, color='mediumslateblue', linewidth=2, label='Spot Price')
plt.xlabel('Time [hour]', fontsize=14)
plt.ylabel('Spot Price [NOK/kWh]', fontsize=14)
plt.title('Spot Prices: Modified Scenario', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('/Users/kamillamoen/Documents/plots_latex/spot_price_plot_modified.eps', format='eps', dpi=300)  # Lagre som PDF
plt.show()

plt.figure(figsize=(10, 6))
plt.grid(color='lightgrey', linestyle='--', linewidth=0.7)
plt.plot(np.arange(len(modified_prices)), np.sort(modified_prices)[::-1], color='cornflowerblue', linewidth=3, label='Sorted Spot Price')
plt.fill_between(np.arange(len(modified_prices)), np.sort(modified_prices)[::-1], color='lightsteelblue', alpha=0.6)
plt.xlabel('Time [hour]', fontsize=14)
plt.ylabel('Spot Price [NOK/kWh]', fontsize=14)
plt.title('Spot prices sorted', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('/Users/kamillamoen/Documents/plots_latex/spot_price_plot_sorted_modified.eps', format='eps', dpi=300)  # Lagre som PDF
plt.show()









'''



def generate_random_numbers(total_numbers = 8760, zero_percentage = 0.3):
    # Antall nuller
    num_zeros = int(total_numbers * zero_percentage)
    zero_list = [0 for i in range(num_zeros)]

    # Generer tilfeldige tall mellom 1 og 100
    random_numbers = np.random.randint(1, 101, size = total_numbers - num_zeros)

    # Plasser nuller i listen
    result = np.concatenate([random_numbers, zero_list])

    # Bland tallene
    np.random.shuffle(result)

    return result


def adjust_prices_based_on_time_and_season(prices):
    # Justeringer for tid på døgnet og årstid
    adjusted_prices = np.copy(prices)

    for hour in range(24):
        for day in range(365):
            index = hour + day * 24
            # Høyere priser om dagen (08:00 til 20:00)
            if 8 <= hour < 20:
                adjusted_prices[index] *= 1.5  # 50% høyere om dagen
            # Høyere priser om vinteren (desember, januar, februar)
            if (day % 365) // 30 in [11, 0, 1]:  # Desember, januar, februar
                adjusted_prices[index] *= 2  # 30% høyere om vinteren
            elif (day % 365) // 30 in [10, 9, 2, 3]:  # Desember, januar, februar
                adjusted_prices[index] *= 1.5  # 30% høyere om vinteren

    return adjusted_prices


def normalize_prices(prices, target_mean):
    # Normaliser priser slik at gjennomsnittet blir lik target_mean
    current_mean = np.mean(prices)
    adjustment_factor = target_mean / current_mean
    normalized_prices = prices * adjustment_factor
    return normalized_prices


# Generer tallene
random_numbers = generate_random_numbers()
print(len(random_numbers))

# Juster priser basert på tid på døgnet og årstid
adjusted_prices = adjust_prices_based_on_time_and_season(random_numbers)

# Normaliser prisene for å sikre at gjennomsnittet er lik 50
normalized_prices = normalize_prices(adjusted_prices, target_mean = 50)

# Lag en DataFrame for lagring i CSV
df = pd.DataFrame({'Hour': np.arange(8760),  # Time
                   'Price': normalized_prices})

# Lagre til CSV-fil
df.to_csv('new.csv', index = False)

# Sjekk gjennomsnittet og antall nuller
print("Gjennomsnitt etter normalisering:", np.mean(normalized_prices))
print("Antall nuller:", np.sum(normalized_prices == 0))
print("Data lagret i 'new.csv'")


# Konverter listen til en NumPy-array for enklere beregning
prices_array = np.array(normalized_prices)

# Beregn gjennomsnittet per døgn
daily_average_prices = np.mean(prices_array.reshape(365, 24), axis=1)

# Konverter tilbake til liste hvis ønskelig
daily_average_prices_list = daily_average_prices.tolist()


plt.plot(np.arange(365), daily_average_prices_list)
plt.show()'''