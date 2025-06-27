import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def chi_square_uniformity_test(data, bins=10, alpha=0.05):
    """
    Perform chi-square test for uniformity on a series of random numbers.

    Parameters:
    - data: array-like, the random numbers to test
    - bins: int, number of bins to divide the data into
    - alpha: float, significance level (default 0.05)

    Returns:
    - Dictionary with test results and interpretation
    """
    data = np.array(data)
    n = len(data)

    # Histogram (observed frequency)
    hist, bin_edges = np.histogram(data, bins=bins)
    observed = hist

    # Expected frequency for uniform distribution
    expected = n / bins
    expected_array = np.full(bins, expected)

    # Chi-square statistic
    chi2_stat = np.sum((observed - expected_array)**2 / expected_array)
    degrees_of_freedom = bins - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, degrees_of_freedom)
    critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
    passes_test = chi2_stat < critical_value

    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'degrees_of_freedom': degrees_of_freedom,
        'observed_frequencies': observed,
        'expected_frequency': expected,
        'passes_test': passes_test,
        'alpha': alpha,
        'bin_edges': bin_edges,
        'interpretation': interpret_results(passes_test, p_value, alpha, chi2_stat, critical_value)
    }

def interpret_results(passes_test, p_value, alpha, chi2_stat, critical_value):
    """Provide human-readable interpretation of chi-square test results."""
    if passes_test:
        return {
            'conclusion': "PASSES - Data appears uniformly distributed",
            'reason': f"Chi-square = {chi2_stat:.4f} < Critical value = {critical_value:.4f}",
            'statistical_meaning': f"p-value = {p_value:.4f} > α = {alpha}, fail to reject H₀"
        }
    else:
        return {
            'conclusion': "FAILS - Data is NOT uniformly distributed",
            'reason': f"Chi-square = {chi2_stat:.4f} > Critical value = {critical_value:.4f}",
            'statistical_meaning': f"p-value = {p_value:.4f} ≤ α = {alpha}, reject H₀"
        }

def visualize_test_results(data, test_results):
    """Visualize observed vs expected frequencies and distribution histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bins = len(test_results['observed_frequencies'])
    x_pos = np.arange(bins)

    ax1.bar(x_pos - 0.2, test_results['observed_frequencies'], 0.4, label='Observed', color='skyblue')
    ax1.bar(x_pos + 0.2, [test_results['expected_frequency']] * bins, 0.4, label='Expected', color='salmon')
    ax1.set_title("Observed vs Expected Frequencies")
    ax1.set_xlabel("Bin Index")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(data, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax2.set_title("Histogram of Random Data")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def print_results(results):
    """Print test outcome in a readable format."""
    print(f"Chi-square statistic: {results['chi2_statistic']:.4f}")
    print(f"Critical value: {results['critical_value']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Degrees of freedom: {results['degrees_of_freedom']}")
    print(f"\nResult: {results['interpretation']['conclusion']}")
    print(f"Reason: {results['interpretation']['reason']}")
    print(f"Statistical meaning: {results['interpretation']['statistical_meaning']}")
    print(f"\nObserved frequencies: {results['observed_frequencies']}")
    print(f"Expected frequency per bin: {results['expected_frequency']:.2f}\n")

def generate_test_examples():
    print("CHI-SQUARE TEST FOR UNIFORMITY\n" + "="*40)

    # Good uniform distribution
    print("\nExample 1: Uniform Distribution (should PASS)")
    np.random.seed(42)
    good_data = np.random.uniform(0, 1, 1000)
    res1 = chi_square_uniformity_test(good_data)
    print_results(res1)
    visualize_test_results(good_data, res1)

    # Biased (non-uniform) distribution
    print("\nExample 2: Biased Distribution (Beta(5,5), should FAIL)")
    np.random.seed(42)
    biased_data = np.random.beta(5, 5, 1000)
    res2 = chi_square_uniformity_test(biased_data)
    print_results(res2)
    visualize_test_results(biased_data, res2)

    # Extremely skewed distribution
    print("\nExample 3: Severely Non-uniform (80% in first 30%)")
    np.random.seed(42)
    non_uniform_data = np.concatenate([
        np.random.uniform(0, 0.3, 800),
        np.random.uniform(0.3, 1, 200)
    ])
    np.random.shuffle(non_uniform_data)
    res3 = chi_square_uniformity_test(non_uniform_data)
    print_results(res3)
    visualize_test_results(non_uniform_data, res3)

if __name__ == "__main__":
    # Use chi-square test on N random numbers from 1 to 100
    N = 10
    #np.random.seed(42) # 123345
    np.random.seed()
    x = np.random.uniform(1, 100, N)
    print(f"\nCustom Test: {N} Random Numbers from Uniform(1,100)")
    result_custom = chi_square_uniformity_test(x)
    print_results(result_custom)
    visualize_test_results(x, result_custom)

    # Run additional examples
    generate_test_examples()

"""
Custom Test: 10 Random Numbers from Uniform(1,100)
Chi-square statistic: 16.0000
Critical value: 16.9190
p-value: 0.0669
Degrees of freedom: 9

Result: PASSES - Data appears uniformly distributed
Reason: Chi-square = 16.0000 < Critical value = 16.9190
Statistical meaning: p-value = 0.0669 > α = 0.05, fail to reject H₀

Observed frequencies: [2 0 0 0 0 2 0 1 1 4]
Expected frequency per bin: 1.00

![chi_squared_test_00](chi_squared_test_00.png)

CHI-SQUARE TEST FOR UNIFORMITY
========================================

Example 1: Uniform Distribution (should PASS)
Chi-square statistic: 10.0000
Critical value: 16.9190
p-value: 0.3505
Degrees of freedom: 9

Result: PASSES - Data appears uniformly distributed
Reason: Chi-square = 10.0000 < Critical value = 16.9190
Statistical meaning: p-value = 0.3505 > α = 0.05, fail to reject H₀

Observed frequencies: [114 112  95 102  81 111  98  88 100  99]
Expected frequency per bin: 100.00

![chi_squared_test_01](chi_squared_test_01.png)

Example 2: Biased Distribution (Beta(5,5), should FAIL)
Chi-square statistic: 514.4200
Critical value: 16.9190
p-value: 0.0000
Degrees of freedom: 9

Result: FAILS - Data is NOT uniformly distributed
Reason: Chi-square = 514.4200 > Critical value = 16.9190
Statistical meaning: p-value = 0.0000 ≤ α = 0.05, reject H₀

Observed frequencies: [  3  26  74 153 193 203 175  95  58  20]
Expected frequency per bin: 100.00

![chi_squared_test_02](chi_squared_test_02.png)

Example 3: Severely Non-uniform (80% in first 30%)
Chi-square statistic: 1193.0400
Critical value: 16.9190
p-value: 0.0000
Degrees of freedom: 9

Result: FAILS - Data is NOT uniformly distributed
Reason: Chi-square = 1193.0400 > Critical value = 16.9190
Statistical meaning: p-value = 0.0000 ≤ α = 0.05, reject H₀

Observed frequencies: [276 264 260  33  35  29  26  29  22  26]
Expected frequency per bin: 100.00

![chi_squared_test_03](chi_squared_test_03.png)
"""