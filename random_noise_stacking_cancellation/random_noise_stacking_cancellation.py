import numpy as np
import matplotlib.pyplot as plt

# Function to generate a random series of strictly 1 or -1
def generate_random_series(length):
    return np.random.choice([-1, 1], size=length)

# Function to stack series additively and normalize
def stack_and_normalize_series(num_series, length):
    cumulative_series = np.zeros(length)
    for _ in range(num_series):
        cumulative_series += generate_random_series(length)
    return cumulative_series / num_series

# Function to compute Euclidean distance
def compute_euclidean_distance(series):
    return np.linalg.norm(series)

# Parameters
num_series_list = [1, 5, 10, 50, 100, 500, 1000, 100000, 10000000]
length = 1000

# Generate, stack, and normalize series
distances = []
for num_series in num_series_list:
    normalized_series = stack_and_normalize_series(num_series, length)
    distance = compute_euclidean_distance(normalized_series)
    distances.append(distance)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_series_list, distances, marker='o')
plt.xlabel('Number of Series Stacked')
plt.ylabel('Euclidean Distance')
plt.title('Euclidean Distance of Normalized Stacked Random Series')
plt.grid(True)
plt.show()

# Demonstrate noise cancellation
plt.figure(figsize=(12, 8))
for i, num_series in enumerate(num_series_list):
    normalized_series = stack_and_normalize_series(num_series, length)
    plt.subplot(len(num_series_list)//2 + 1, 2, i + 1)
    plt.plot(normalized_series)
    plt.title(f'{num_series} Series Stacked')
plt.tight_layout()
plt.show()

"""
1. Generate random series 
2. Stack them addictively and show random noise cancelling 
3. Plot to demo 
4. Use Euclidean distance to prove it goes to zero
"""