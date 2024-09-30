import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_random_numbers(total_numbers = 8760, target_mean = 50, zero_percentage = 0.3):
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
plt.show()