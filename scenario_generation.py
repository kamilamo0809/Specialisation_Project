import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Load the 2023 spot prices from CSV
data = pd.read_excel("spotpriser_23.xlsx")
prices_2023 = data['NO1'].values

zero_negative_ratio = 0.115  # 30% zero or negative prices
target_mean = 0.6  # Desired mean price

# Count how many zero or negative prices are already in the 2023 data
num_zero_negative_existing = np.sum(prices_2023 <= 0)

# Calculate how many more zero or negative prices we need
num_prices = len(prices_2023)
num_zero_negative_needed = int(zero_negative_ratio * num_prices)
num_zero_negative_to_add = num_zero_negative_needed - num_zero_negative_existing

# Randomly select the indices to be set to zero or negative values
if num_zero_negative_to_add > 0:
    zero_negative_indices = np.random.choice(np.where(prices_2023 > 0)[0], size=num_zero_negative_to_add, replace=False)
else:
    zero_negative_indices = []

# Scale prices around the pattern of 2023 prices
# Normalize the original prices so that their mean is 0 (zero-centered)
mean_2023 = np.mean(prices_2023)
scaled_prices = prices_2023 - mean_2023

# Add a target mean
scaled_prices += target_mean

# Set the required number of prices to zero or negative values
modified_prices = scaled_prices.copy()
if num_zero_negative_to_add > 0:
    modified_prices[zero_negative_indices] = np.random.uniform(low=-1.5, high=0, size=num_zero_negative_to_add)

# Verify the mean and pattern
print(f"New mean price: {np.mean(modified_prices)}")
print(f"Number of zero or negative prices: {np.sum(modified_prices <= 0)}, precantage: {round(np.sum(modified_prices <= 0)/8760 * 100, 1)}%")
no_prod = sum(1 for i in modified_prices if i < 0.1)
print('Number of hours where the spot price is lower than the marginal cost of production: ', no_prod)

# Create a DataFrame with 'Hour' as the index and 'Price' as the price column
df = pd.DataFrame({
    'Hour': range(1, len(modified_prices) + 1),  # Assuming you want hours from 1 to n
    'Price': modified_prices
})
df.to_csv("modified_spot_prices.csv", index=False)


# Plot
plt.figure(figsize=(10, 6))
plt.grid(color='lightgrey', linestyle='--', linewidth=0.7)
plt.plot(modified_prices, color='mediumslateblue', linewidth=2, label='Spot Price')
plt.xlabel('Time [hour]', fontsize=14)
plt.ylabel('Spot Price [NOK/kWh]', fontsize=14)
plt.title('Spot Prices: Modified Scenario', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
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
plt.show()
