"""
This script performs the following tasks for analyzing fire damage claims and reinsurance impact:

1. **Plot Histograms**:
   - Generates and saves two histograms of claim amounts:
     - One with normal frequency.
     - One with a log-scaled frequency axis.
   - Saves the figures as `histogram_normal.png` and `histogram_log.png` for the LaTeX report.

2. **Simulate Claims and Compare to Theory**:
   - Simulates total claims using Poisson-distributed claim frequency and empirical claim severity.
   - Compares simulated total losses with theoretical expected values.
   - Evaluates whether the theoretical calculations match empirical results.

3. **Reinsurance Impact Analysis**:
   - Applies Excess-of-Loss (XL) reinsurance formulas to simulated claims.
   - Computes total expected loss and variance under different reinsurance contracts.
   - Compares the effect of reinsurance on financial risk.

4. **Generate Quantile Comparison Table**:
   - Computes and displays key quantiles of total loss distributions:
     - No reinsurance vs. two XL reinsurance contracts.
   - Saves the table for use in the LaTeX report.

5. **Additional Visualizations**:
   - QQ plot comparing total losses to normal distribution
   - Boxplot comparing reinsurance scenarios
   - Density plot comparing distributions for different reinsurance contracts
   - Risk reduction comparison table

All figures and tables will be saved as PNG or CSV files where applicable for inclusion in the report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for simulation
NUM_SIMULATIONS = 50000
POISSON_LAMBDA = 567  # Expected number of claims per year
QUANTILES = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]  # Quantiles to compute
FIGSIZE = (14, 8)  # Standard figure size for all plots

# Reinsurance parameters
M1, L1 = 10, 40  # First contract: M = 10M, L = 40M (capped)
M2, L2 = 10, np.inf  # Second contract: M = 10M, L = ∞ (no upper cap)

def load_data():
    """Load the Danish fire damage data."""
    data = pd.read_csv("danishDataFinal.csv")
    # Convert Loss to numeric if it's not already
    data["Loss"] = pd.to_numeric(data["Loss"])
    return data

def plot_histograms(data):
    """
    Create and save histograms of claim amounts split into two ranges:
    1. Non-extreme values (0-10M)
    2. Extreme values (>10M)
    """
    # Use the same number of bins for both histograms
    num_bins = 40
    
    # Create separate histograms for different ranges
    # 1. Non-extreme values (0-10M)
    plt.figure(figsize=FIGSIZE)
    non_extreme = data[data["Loss"] <= 10]
    sns.histplot(non_extreme["Loss"], kde=True, bins=num_bins)
    plt.title("Histogram of Non-Extreme Fire Damage Claims (0-10M)")
    plt.xlabel("Claim Amount (in millions)")
    plt.ylabel("Frequency")
    plt.xlim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.savefig("histogram_normal.png", dpi=300, bbox_inches="tight")
    
    # 2. Extreme values (>10M)
    plt.figure(figsize=FIGSIZE)
    extreme = data[data["Loss"] > 10]
    sns.histplot(extreme["Loss"], kde=True, bins=num_bins)
    plt.title("Histogram of Extreme Fire Damage Claims (>10M)")
    plt.xlabel("Claim Amount (in millions)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig("histogram_extreme.png", dpi=300, bbox_inches="tight")
    
    print("Histograms created and saved.")
    
    # Additional summary statistics
    print(f"Number of claims: {len(data)}")
    print(f"Number of claims ≤ 10M: {len(non_extreme)} ({len(non_extreme)/len(data):.1%})")
    print(f"Number of claims > 10M: {len(extreme)} ({len(extreme)/len(data):.1%})")
    print(f"Maximum claim amount: {data['Loss'].max():.2f}M")

def find_best_distribution(data):
    """
    Find the best-fitting distribution for the data.
    
    Args:
        data: Array of values to test
        
    Returns:
        tuple: (best_distribution_name, fitted_params)
    """
    # List of distributions to test
    distributions = [
        stats.gamma, stats.pareto, stats.lognorm, stats.weibull_min, 
        stats.expon, stats.genpareto
    ]
    
    best_distribution = None
    best_params = None
    best_sse = np.inf
    
    # Test each distribution
    for distribution in distributions:
        try:
            # Fit distribution
            params = distribution.fit(data)
            
            # Calculate goodness of fit (Sum of Squared Errors)
            cdf_fitted = distribution.cdf(sorted(data), *params)
            empirical_cdf = np.arange(1, len(data) + 1) / len(data)
            sse = np.sum((cdf_fitted - empirical_cdf) ** 2)
            
            # Select best distribution
            if sse < best_sse:
                best_distribution = distribution
                best_params = params
                best_sse = sse
        except Exception as e:
            print(f"Error fitting {distribution.name}: {str(e)}")
            continue
    
    if best_distribution is None:
        print("Warning: Could not find a good distribution fit. Using normal as fallback.")
        return stats.norm, stats.norm.fit(data)
    
    return best_distribution, best_params

def create_qq_plot(data, filename="qq_plot.png"):
    """
    Create a QQ plot comparing the empirical distribution to the best-fitting distribution.
    
    Args:
        data: Array of values to plot
        filename: Name of the file to save the plot
    """
    print("Finding best-fitting distribution for QQ plot...")
    best_dist, params = find_best_distribution(data)
    
    plt.figure(figsize=FIGSIZE)
    
    # Create QQ plot with the best-fitting distribution
    # Due to the way stats.probplot works, we'll use the normal QQ plot and transform our data
    if best_dist.name == 'norm':
        stats.probplot(data, dist="norm", plot=plt)
    else:
        # Transform data according to the best-fitting distribution
        transformed_data = stats.norm.ppf(best_dist.cdf(data, *params))
        stats.probplot(transformed_data, dist="norm", plot=plt)
    
    plt.title(f"QQ Plot: Hybrid-Simulated Total Claims vs Fitted {best_dist.name.capitalize()}")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"QQ plot created with {best_dist.name} distribution and saved as {filename}.")
    
    # Also create a CDF comparison plot
    plt.figure(figsize=FIGSIZE)
    
    # Calculate empirical CDF
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    
    # Plot empirical CDF
    plt.plot(x, y, 'b-', label='Hybrid-Simulated Claims CDF')
    
    # Plot fitted CDF
    x_fit = np.linspace(min(data), max(data), 1000)
    y_fit = best_dist.cdf(x_fit, *params)
    plt.plot(x_fit, y_fit, 'r-', label=f'Fitted {best_dist.name.capitalize()} CDF')
    
    plt.title(f"CDF Comparison: Hybrid-Simulated Claims vs Fitted {best_dist.name.capitalize()}")
    plt.xlabel("Total Claim Amount (in millions)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("cdf_comparison.png", dpi=300, bbox_inches="tight")
    print(f"CDF comparison plot created and saved.")

def plot_reinsurance_comparisons(total_claims, reinsured_claims1, reinsured_claims2):
    """
    Create visualization comparing the different reinsurance scenarios.
    
    Args:
        total_claims: Array of simulated total claims (no reinsurance)
        reinsured_claims1: Array of claims under first reinsurance contract (M=10, L=40)
        reinsured_claims2: Array of claims under second reinsurance contract (M=10, L=∞)
    """
    # Create a boxplot comparison
    plt.figure(figsize=FIGSIZE)
    # Prepare data for boxplot
    box_data = pd.DataFrame({
        'No Reinsurance (Hybrid Model)': total_claims,
        'XL (M=10, L=40)': reinsured_claims1,
        'XL (M=10, L=∞)': reinsured_claims2
    })
    
    # Check if third scenario (M=10, L=∞) has any variation
    if np.std(reinsured_claims2) < 0.0001:
        print(f"Note: All values in XL (M=10, L=∞) are approximately {reinsured_claims2[0]:.2f}.")
        print("This is expected behavior when retention limit M=10 and no upper cap (L=∞).")
        print("The insurer retains only M=10 for each claim, with the rest transferred to reinsurance.")
    
    sns.boxplot(data=box_data)
    plt.title("Boxplot Comparison of Reinsurance Scenarios (Hybrid Claim Model)")
    plt.ylabel("Total Claim Amount (in millions)")
    plt.grid(True, alpha=0.3)
    plt.savefig("reinsurance_boxplot.png", dpi=300, bbox_inches="tight")
    print("Reinsurance boxplot created and saved.")
    
    # Create a density plot comparison
    plt.figure(figsize=FIGSIZE)
    
    # Plot first two scenarios normally
    sns.kdeplot(total_claims, label='No Reinsurance (Hybrid Model)', fill=True, alpha=0.3)
    sns.kdeplot(reinsured_claims1, label='XL (M=10, L=40)', fill=True, alpha=0.3)
    
    # For the third scenario, if it's all constant values, add a vertical line
    if np.std(reinsured_claims2) < 0.0001:
        plt.axvline(x=reinsured_claims2[0], color='g', linestyle='--', 
                  label=f'XL (M=10, L=∞): All values = {reinsured_claims2[0]:.2f}')
    else:
        # In case there is variation
        sns.kdeplot(reinsured_claims2, label='XL (M=10, L=∞)', fill=True, alpha=0.3)
    
    plt.title("Density Comparison of Reinsurance Scenarios (Hybrid Claim Model)")
    plt.xlabel("Total Claim Amount (in millions)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reinsurance_density.png", dpi=300, bbox_inches="tight")
    print("Reinsurance density plot created and saved.")

def create_risk_metrics_table(reinsurance_results):
    """
    Create a table of risk metrics for different reinsurance scenarios.
    
    Args:
        reinsurance_results: Results from analyze_reinsurance_impact
        
    Returns:
        DataFrame: Table of risk metrics
    """
    # Create DataFrame with risk metrics
    data = {
        "Metric": ["Expected Value", "Standard Deviation", "Coefficient of Variation", 
                  "99% VaR", "Interquartile Range"],
        "No Reinsurance": [
            reinsurance_results["no_reinsurance"]["expected_value"],
            reinsurance_results["no_reinsurance"]["std_deviation"],
            reinsurance_results["no_reinsurance"]["std_deviation"] / reinsurance_results["no_reinsurance"]["expected_value"],
            np.quantile(reinsurance_results["no_reinsurance"]["quantiles"], 0.99),
            np.quantile(reinsurance_results["no_reinsurance"]["quantiles"], 0.75) - np.quantile(reinsurance_results["no_reinsurance"]["quantiles"], 0.25)
        ],
        "XL (M=10, L=40)": [
            reinsurance_results["reinsurance_capped"]["expected_value"],
            reinsurance_results["reinsurance_capped"]["std_deviation"],
            reinsurance_results["reinsurance_capped"]["std_deviation"] / reinsurance_results["reinsurance_capped"]["expected_value"],
            np.quantile(reinsurance_results["reinsurance_capped"]["quantiles"], 0.99),
            np.quantile(reinsurance_results["reinsurance_capped"]["quantiles"], 0.75) - np.quantile(reinsurance_results["reinsurance_capped"]["quantiles"], 0.25)
        ],
        "XL (M=10, L=∞)": [
            reinsurance_results["reinsurance_uncapped"]["expected_value"],
            reinsurance_results["reinsurance_uncapped"]["std_deviation"],
            reinsurance_results["reinsurance_uncapped"]["std_deviation"] / (reinsurance_results["reinsurance_uncapped"]["expected_value"] + 1e-10),  # Avoid division by zero
            np.quantile(reinsurance_results["reinsurance_uncapped"]["quantiles"], 0.99),
            np.quantile(reinsurance_results["reinsurance_uncapped"]["quantiles"], 0.75) - np.quantile(reinsurance_results["reinsurance_uncapped"]["quantiles"], 0.25)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Add a note about the third scenario
    if reinsurance_results["reinsurance_uncapped"]["std_deviation"] < 0.0001:
        print("Note: XL (M=10, L=∞) shows no variation because with unlimited cap,")
        print("the insurer retains only the retention limit M=10 for each claim.")
    
    # Save to CSV for LaTeX
    df.to_csv("risk_metrics_comparison.csv", index=False)
    
    return df

def simulate_total_claims(data, num_simulations=NUM_SIMULATIONS, lambda_=POISSON_LAMBDA):
    """
    Simulate total claim amounts using:
    - Poisson distribution for claim frequency
    - Empirical distribution for claim severity
    
    Returns:
        Tuple containing:
        - Array of simulated total claim amounts
        - List of lists, where each inner list contains the individual claims for a simulation
    """
    claims_data = data["Loss"].values
    
    # Store all simulated total claims
    total_claims = []
    # Store all individual claims for each simulation
    all_individual_claims = []
    
    for _ in range(num_simulations):
        # Simulate number of claims using Poisson distribution
        num_claims = np.random.poisson(lambda_)
        
        # Sample from empirical claim severity distribution
        if num_claims > 0:
            # Sample with replacement from the empirical distribution
            claim_amounts = np.random.choice(claims_data, size=num_claims, replace=True)
            total_claim = np.sum(claim_amounts)
            # Store individual claims for this simulation
            all_individual_claims.append(claim_amounts)
        else:
            total_claim = 0
            # No claims for this simulation
            all_individual_claims.append(np.array([]))
            
        total_claims.append(total_claim)
    
    return np.array(total_claims), all_individual_claims

def calculate_theoretical_values(data, lambda_=POISSON_LAMBDA):
    """
    Calculate theoretical expected value and variance of total claims.
    
    Returns:
        dict: Theoretical expected value and variance
    """
    # For total claims S = X1 + X2 + ... + XN where N ~ Poisson(lambda)
    # E[S] = E[N] * E[X]
    # Var[S] = E[N] * Var[X] + Var[N] * (E[X])²
    
    claims_data = data["Loss"].values
    
    # Calculate statistics of individual claims
    expected_x = np.mean(claims_data)
    var_x = np.var(claims_data)
    
    # Poisson has mean = variance = lambda
    expected_n = lambda_
    var_n = lambda_
    
    # Calculate theoretical values for total claims
    expected_s = expected_n * expected_x
    var_s = expected_n * var_x + var_n * (expected_x ** 2)
    
    return {
        "expected_value": expected_s,
        "variance": var_s,
        "std_deviation": np.sqrt(var_s)
    }

def apply_xl_reinsurance_to_individual(claim, M, L):
    """
    Apply Excess-of-Loss (XL) reinsurance to an individual claim.
    
    Args:
        claim: Individual claim amount
        M: Retention limit (insurer's retention)
        L: Upper limit (ceding limit)
        
    Returns:
        float: Net claim retained by the insurer
    """
    if L == np.inf:
        # With unlimited cover, insurer retains min(claim, M)
        return min(claim, M)
    else:
        # With limited cover, insurer retains min(claim, M) + max(0, claim - (M+L))
        return min(claim, M) + max(0, claim - (M + L))

def plot_reinsurance_revenue(total_claims, reinsured_claims1, reinsured_claims2):
    """
    Plot the density and quantile functions of reinsurance revenue.
    
    Revenue = Premium Income - Total Claims Cost - Reinsurance Premium
    
    Args:
        total_claims: Array of simulated total claims (no reinsurance)
        reinsured_claims1: Array of claims under first reinsurance contract (M=10, L=40)
        reinsured_claims2: Array of claims under second reinsurance contract (M=10, L=∞)
        
    Returns:
        DataFrame with revenue statistics and quantiles
    """
    # Parameters from the problem
    num_policies = 630000
    premium_per_policy = 3331  # DKK
    reinsurance_premium1 = 609  # DKK per policy for Contract 1
    reinsurance_premium2 = 872  # DKK per policy for Contract 2
    
    # Calculate total premium income and reinsurance premiums
    total_premium = num_policies * premium_per_policy / 1e6  # Convert to millions
    total_reinsurance_premium1 = num_policies * reinsurance_premium1 / 1e6  # Convert to millions
    total_reinsurance_premium2 = num_policies * reinsurance_premium2 / 1e6  # Convert to millions
    
    # Double check calculations - print the values for verification
    print(f"\nVerifying premium calculations:")
    print(f"Total premium income: {total_premium:.2f} million DKK")
    print(f"Reinsurance premium (Contract 1): {total_reinsurance_premium1:.2f} million DKK")
    print(f"Reinsurance premium (Contract 2): {total_reinsurance_premium2:.2f} million DKK")
    
    # Calculate revenue for each scenario
    revenue_no_reinsurance = total_premium - total_claims
    revenue_with_reinsurance1 = total_premium - reinsured_claims1 - total_reinsurance_premium1
    revenue_with_reinsurance2 = total_premium - reinsured_claims2 - total_reinsurance_premium2
    
    # Create density plot of revenues
    plt.figure(figsize=FIGSIZE)
    sns.kdeplot(revenue_no_reinsurance, label='No Reinsurance', fill=True, alpha=0.3)
    sns.kdeplot(revenue_with_reinsurance1, label='XL (M=10, L=40), Premium=609 DKK', fill=True, alpha=0.3)
    sns.kdeplot(revenue_with_reinsurance2, label='XL (M=10, L=∞), Premium=872 DKK', fill=True, alpha=0.3)
    
    plt.title("Density of Insurance Revenue with Different Reinsurance Contracts")
    plt.xlabel("Revenue (in millions DKK)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("reinsurance_revenue_density.png", dpi=300, bbox_inches="tight")
    print("Reinsurance revenue density plot created and saved.")
    
    # Create a better quantile plot with simplified spacing
    plt.figure(figsize=FIGSIZE)
    
    # Use consistent intervals of 0.1% (0.001) as requested
    quantiles = np.linspace(0.002, 0.998, 499)  # From 0.1% to 99.9% in steps of 0.1%
    
    # Calculate quantiles for all three scenarios
    revenue_no_reinsurance_quantiles = np.quantile(revenue_no_reinsurance, quantiles)
    revenue_with_reinsurance1_quantiles = np.quantile(revenue_with_reinsurance1, quantiles)
    revenue_with_reinsurance2_quantiles = np.quantile(revenue_with_reinsurance2, quantiles)
    
    # Plot quantiles with added transparency
    plt.plot(quantiles, revenue_no_reinsurance_quantiles, label='No Reinsurance', alpha=0.7)
    plt.plot(quantiles, revenue_with_reinsurance1_quantiles, label='XL (M=10, L=40), Premium=609 DKK', alpha=0.7)
    plt.plot(quantiles, revenue_with_reinsurance2_quantiles, label='XL (M=10, L=∞), Premium=872 DKK', alpha=0.7)
    
    # Add labels and formatting
    plt.title("Quantile Plot of Insurance Revenue with Different Reinsurance Contracts")
    plt.xlabel("Quantile")
    plt.ylabel("Revenue (in millions DKK)")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis limit to cut off extreme negative values
    plt.ylim(bottom=-1000)
    
    # Save the plot
    plt.savefig("reinsurance_revenue_quantile.png", dpi=300, bbox_inches="tight")
    print("Simplified reinsurance revenue quantile plot created and saved.")
    
    # Create a standard table with key revenue statistics
    # Calculate loss ratios
    mean_total_claims = np.mean(total_claims)
    mean_reinsured_claims1 = np.mean(reinsured_claims1)
    mean_reinsured_claims2 = np.mean(reinsured_claims2)
    
    loss_ratio_no_reinsurance = mean_total_claims / total_premium * 100
    loss_ratio_reinsurance1 = (mean_reinsured_claims1 + total_reinsurance_premium1) / total_premium * 100
    loss_ratio_reinsurance2 = (mean_reinsured_claims2 + total_reinsurance_premium2) / total_premium * 100
    
    standard_data = {
        "Metric": ["Expected Revenue", "Std Deviation", "CV", "Loss Ratio (%)", "5% VaR (Loss)", "5% CTE (Loss)","1% VaR (Loss)", "1% CTE (Loss)"],
        "No Reinsurance": [
            np.mean(revenue_no_reinsurance),
            np.std(revenue_no_reinsurance),
            np.std(revenue_no_reinsurance) / np.mean(revenue_no_reinsurance),
            loss_ratio_no_reinsurance,
            np.quantile(revenue_no_reinsurance, 0.05),
            np.mean(revenue_no_reinsurance[revenue_no_reinsurance <= np.quantile(revenue_no_reinsurance, 0.05)]),
            np.quantile(revenue_no_reinsurance, 0.01),
            np.mean(revenue_no_reinsurance[revenue_no_reinsurance <= np.quantile(revenue_no_reinsurance, 0.01)])
        ],
        "XL (M=10, L=40)": [
            np.mean(revenue_with_reinsurance1),
            np.std(revenue_with_reinsurance1),
            np.std(revenue_with_reinsurance1) / np.mean(revenue_with_reinsurance1),
            loss_ratio_reinsurance1,
            np.quantile(revenue_with_reinsurance1, 0.05),
            np.mean(revenue_with_reinsurance1[revenue_with_reinsurance1 <= np.quantile(revenue_with_reinsurance1, 0.05)]),
            np.quantile(revenue_with_reinsurance1, 0.01),
            np.mean(revenue_with_reinsurance1[revenue_with_reinsurance1 <= np.quantile(revenue_with_reinsurance1, 0.01)])
        ],
        "XL (M=10, L=∞)": [
            np.mean(revenue_with_reinsurance2),
            np.std(revenue_with_reinsurance2),
            np.std(revenue_with_reinsurance2) / np.mean(revenue_with_reinsurance2),
            loss_ratio_reinsurance2,
            np.quantile(revenue_with_reinsurance2, 0.05),
            np.mean(revenue_with_reinsurance2[revenue_with_reinsurance2 <= np.quantile(revenue_with_reinsurance2, 0.05)]),
            np.quantile(revenue_with_reinsurance2, 0.01),
            np.mean(revenue_with_reinsurance2[revenue_with_reinsurance2 <= np.quantile(revenue_with_reinsurance2, 0.01)])
        ]
    }
    
    df_standard = pd.DataFrame(standard_data)
    # Save to CSV for LaTeX
    df_standard.to_csv("revenue_comparison.csv", index=False)
    print("Revenue comparison table created and saved.")
    
    # Print expected values for verification
    print(f"\nExpected total claims (No Reinsurance): {mean_total_claims:.2f} million DKK")
    print(f"Expected retained claims (XL, M=10, L=40): {mean_reinsured_claims1:.2f} million DKK")
    print(f"Expected retained claims (XL, M=10, L=∞): {mean_reinsured_claims2:.2f} million DKK")
    
    print(f"Expected revenue (No Reinsurance): {np.mean(revenue_no_reinsurance):.2f} million DKK")
    print(f"Expected revenue (XL, M=10, L=40): {np.mean(revenue_with_reinsurance1):.2f} million DKK")
    print(f"Expected revenue (XL, M=10, L=∞): {np.mean(revenue_with_reinsurance2):.2f} million DKK")
    
    print(f"Loss ratio (No Reinsurance): {loss_ratio_no_reinsurance:.1f}%")
    print(f"Loss ratio (XL, M=10, L=40): {loss_ratio_reinsurance1:.1f}%")
    print(f"Loss ratio (XL, M=10, L=∞): {loss_ratio_reinsurance2:.1f}%")
    
    # Create a table of revenue quantiles for LaTeX
    quantile_points = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
    # Create labels with special formatting for 99.5% and 99.9%
    quantile_labels = []
    for q in quantile_points:
        if q == 0.995:
            quantile_labels.append("99.5%")
        elif q == 0.999:
            quantile_labels.append("99.9%")
        else:
            quantile_labels.append(f"{int(q*100)}%")
    
    revenue_quantiles_data = {
        "Quantile": quantile_labels,
        "No Reinsurance": [np.quantile(revenue_no_reinsurance, q) for q in quantile_points],
        "XL (M=10, L=40)": [np.quantile(revenue_with_reinsurance1, q) for q in quantile_points],
        "XL (M=10, L=∞)": [np.quantile(revenue_with_reinsurance2, q) for q in quantile_points]
    }
    
    df_quantiles = pd.DataFrame(revenue_quantiles_data)
    # Save to CSV for LaTeX
    df_quantiles.to_csv("revenue_quantiles.csv", index=False)
    print("Revenue quantiles table created and saved.")
    
    # Create a table of WORST-CASE revenue scenarios (left tail)
    left_tail_points = [0.0001, 0.001, 0.01, 0.05, 0.1]
    left_tail_labels = ["0.01%", "0.1%", "1%", "5%", "10%"]
    
    worst_case_data = {
        "Quantile": left_tail_labels,
        "No Reinsurance": [np.quantile(revenue_no_reinsurance, q) for q in left_tail_points],
        "XL (M=10, L=40)": [np.quantile(revenue_with_reinsurance1, q) for q in left_tail_points],
        "XL (M=10, L=∞)": [np.quantile(revenue_with_reinsurance2, q) for q in left_tail_points]
    }
    
    df_worst_case = pd.DataFrame(worst_case_data)
    # Save to CSV for LaTeX
    df_worst_case.to_csv("revenue_worst_case.csv", index=False)
    print("Worst-case revenue scenarios table created and saved.")
    
    # Print both tables for immediate reference
    print("\nRevenue Quantiles Table:")
    print(df_quantiles.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    print("\nWorst-Case Revenue Scenarios Table:")
    print(df_worst_case.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    return df_standard, df_quantiles, df_worst_case

def analyze_reinsurance_impact(total_claims, individual_claims):
    """
    Analyze the impact of reinsurance on total losses.
    
    Args:
        total_claims: Array of simulated total claims without reinsurance
        individual_claims: List of lists containing individual claims for each simulation
        
    Returns:
        dict: Statistics for each reinsurance scenario
    """
    # No reinsurance stats (already have total_claims)
    no_reinsurance_stats = {
        "expected_value": np.mean(total_claims),
        "variance": np.var(total_claims),
        "std_deviation": np.std(total_claims),
        "quantiles": np.quantile(total_claims, QUANTILES)
    }
    
    # Apply first reinsurance contract (M = 10, L = 40)
    # For each simulation, apply reinsurance to individual claims and sum
    reinsured_claims1 = []
    for claims in individual_claims:
        if len(claims) > 0:
            # Apply XL reinsurance to each claim in this simulation
            retained_claims = [apply_xl_reinsurance_to_individual(claim, M1, L1-M1) for claim in claims]
            reinsured_claims1.append(np.sum(retained_claims))
        else:
            reinsured_claims1.append(0)
    
    reinsured_claims1 = np.array(reinsured_claims1)
    reinsured_stats1 = {
        "expected_value": np.mean(reinsured_claims1),
        "variance": np.var(reinsured_claims1),
        "std_deviation": np.std(reinsured_claims1),
        "quantiles": np.quantile(reinsured_claims1, QUANTILES)
    }
    
    # Apply second reinsurance contract (M = 10, L = ∞)
    reinsured_claims2 = []
    for claims in individual_claims:
        if len(claims) > 0:
            # With unlimited cover, insurer retains at most M per claim
            retained_claims = [min(claim, M2) for claim in claims]
            reinsured_claims2.append(np.sum(retained_claims))
        else:
            reinsured_claims2.append(0)
    
    reinsured_claims2 = np.array(reinsured_claims2)
    reinsured_stats2 = {
        "expected_value": np.mean(reinsured_claims2),
        "variance": np.var(reinsured_claims2),
        "std_deviation": np.std(reinsured_claims2),
        "quantiles": np.quantile(reinsured_claims2, QUANTILES)
    }
    
    # Create comparison visualizations
    plot_reinsurance_comparisons(total_claims, reinsured_claims1, reinsured_claims2)
    
    # Also plot the reinsurance revenue
    revenue_table, revenue_quantiles, worst_case_revenue = plot_reinsurance_revenue(total_claims, reinsured_claims1, reinsured_claims2)
    print("\nRevenue Statistics:")
    print(revenue_table)
    
    return {
        "no_reinsurance": no_reinsurance_stats,
        "reinsurance_capped": reinsured_stats1,
        "reinsurance_uncapped": reinsured_stats2,
        "reinsured_claims1": reinsured_claims1,
        "reinsured_claims2": reinsured_claims2
    }

def create_quantile_table(reinsurance_results):
    """
    Create a table comparing quantiles across different reinsurance scenarios.
    
    Args:
        reinsurance_results: Results from analyze_reinsurance_impact
        
    Returns:
        DataFrame: Table of quantiles
    """
    # Create DataFrame with quantiles
    quantile_labels = [f"{int(q*100)}%" for q in QUANTILES]
    
    data = {
        "Quantile": quantile_labels,
        "No Reinsurance": reinsurance_results["no_reinsurance"]["quantiles"],
        "XL (M=10, L=40)": reinsurance_results["reinsurance_capped"]["quantiles"],
        "XL (M=10, L=∞)": reinsurance_results["reinsurance_uncapped"]["quantiles"]
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV for LaTeX
    df.to_csv("quantile_comparison.csv", index=False)
    
    return df

def summarize_results(theory, simulation, reinsurance_results):
    """
    Print a summary of the analysis results.
    
    Args:
        theory: Theoretical values
        simulation: Simulated values (mean, variance)
        reinsurance_results: Results from analyzing reinsurance impact
    """
    print("\n=== ANALYSIS RESULTS ===\n")
    
    print("Theoretical vs. Simulated Total Claims:")
    print(f"Theoretical Expected Value: {theory['expected_value']:.2f}")
    print(f"Simulated Expected Value: {simulation['expected_value']:.2f}")
    print(f"Difference: {(simulation['expected_value'] - theory['expected_value']):.2f}")
    print(f"Theoretical Std Deviation: {theory['std_deviation']:.2f}")
    print(f"Simulated Std Deviation: {simulation['std_deviation']:.2f}")
    print(f"Difference: {(simulation['std_deviation'] - theory['std_deviation']):.2f}\n")
    
    print("Impact of Reinsurance:")
    print(f"Expected Loss (No Reinsurance): {reinsurance_results['no_reinsurance']['expected_value']:.2f}")
    print(f"Expected Loss (XL, M=10, L=40): {reinsurance_results['reinsurance_capped']['expected_value']:.2f}")
    print(f"Expected Loss (XL, M=10, L=∞): {reinsurance_results['reinsurance_uncapped']['expected_value']:.2f}\n")
    
    risk_reduction1 = 1 - (reinsurance_results['reinsurance_capped']['std_deviation'] / 
                          reinsurance_results['no_reinsurance']['std_deviation'])
    risk_reduction2 = 1 - (reinsurance_results['reinsurance_uncapped']['std_deviation'] / 
                          reinsurance_results['no_reinsurance']['std_deviation'])
    
    print(f"Risk Reduction (XL, M=10, L=40): {risk_reduction1:.2%}")
    print(f"Risk Reduction (XL, M=10, L=∞): {risk_reduction2:.2%}\n")
    
    # Display the quantile table
    print("Quantile Comparison Table:")
    print(create_quantile_table(reinsurance_results))
    
    # Create and display risk metrics table
    print("\nRisk Metrics Comparison Table:")
    print(create_risk_metrics_table(reinsurance_results))

def simulate_claims_composite(data, num_simulations=NUM_SIMULATIONS, lambda_=567, threshold_percentile=97):
    """
    Simulate claims using a hybrid approach:
    - Use empirical sampling for claims below the threshold
    - Use fitted parametric distribution for claims above the threshold
    
    Args:
        data: DataFrame containing the claims data
        num_simulations: Number of simulations to run
        lambda_: Expected number of claims per year
        threshold_percentile: Percentile threshold for splitting empirical/parametric
        
    Returns:
        Array of simulated total claims
    """
    # Calculate threshold (97th percentile by default)
    claims = data["Loss"].values
    threshold = np.percentile(claims, threshold_percentile)
    print(f"Using hybrid approach with threshold at {threshold:.2f}M (percentile: {threshold_percentile}%)")
    
    # Split claims into non-extreme and extreme
    non_extreme = claims[claims <= threshold]
    extreme = claims[claims > threshold]
    
    # Print summary for the split
    print(f"Non-extreme claims: {len(non_extreme)} ({len(non_extreme)/len(claims):.1%})")
    print(f"Extreme claims: {len(extreme)} ({len(extreme)/len(claims):.1%})")
    
    # Find best distribution for extreme claims
    print("Fitting distribution to extreme claims...")
    best_dist, best_params = find_best_distribution(extreme)
    print(f"Best distribution for extreme claims: {best_dist.name} with parameters: {best_params}")
    
    # Plot the fitted distribution for extreme claims
    plt.figure(figsize=FIGSIZE)
    x = np.linspace(threshold, max(extreme) * 1.1, 1000)
    y = best_dist.pdf(x, *best_params)
    
    # Normalize histogram for comparison with PDF
    plt.hist(extreme, bins=30, density=True, alpha=0.5, label='Extreme Claims')
    plt.plot(x, y, 'r-', lw=2, label=f'Fitted {best_dist.name.capitalize()}')
    
    # Add a vertical line at the threshold
    plt.axvline(x=threshold, color='k', linestyle='--', 
               label=f'Threshold: {threshold:.2f}M ({threshold_percentile}%)')
    
    plt.title(f"Fitted Distribution for Extreme Claims (>{threshold:.2f}M)")
    plt.xlabel("Claim Amount (in millions)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"extreme_claims_fit_{threshold_percentile}.png", dpi=300, bbox_inches="tight")
    
    # Simulate claims
    print("Simulating claims using hybrid approach...")
    total_claims = []
    all_individual_claims = []
    
    for _ in range(num_simulations):
        # Generate Poisson number of claims
        num_claims = np.random.poisson(lambda_)
        
        # For each claim, decide if it's extreme
        p_extreme = len(extreme) / len(claims)
        claim_types = np.random.choice(['non-extreme', 'extreme'], 
                                      size=num_claims, 
                                      p=[1-p_extreme, p_extreme])
        
        # Generate individual claims
        individual_claims = []
        for claim_type in claim_types:
            if claim_type == 'non-extreme':
                # Sample from empirical distribution of non-extreme claims
                claim = np.random.choice(non_extreme)
            else:
                # Sample from fitted distribution for extreme claims
                claim = best_dist.rvs(*best_params)
                # Ensure claim is at least at the threshold
                claim = max(claim, threshold)
                
            individual_claims.append(claim)
        
        # Calculate total claim amount
        total_claim = sum(individual_claims)
        total_claims.append(total_claim)
        all_individual_claims.append(individual_claims)
    
    # Plot the distribution of simulated total claims
    plt.figure(figsize=FIGSIZE)
    sns.histplot(total_claims, kde=True, stat="density", bins=50)
    plt.title("Distribution of Total Claims (Hybrid Model)")
    plt.xlabel("Total Claim Amount (in millions)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.savefig("total_claims_distribution.png", dpi=300, bbox_inches="tight")
    
    print(f"Simulation completed with {num_simulations} runs.")
    print(f"Mean total claim: {np.mean(total_claims):.2f}M")
    print(f"Median total claim: {np.median(total_claims):.2f}M")
    print(f"Std dev of total claims: {np.std(total_claims):.2f}M")
    
    # Also calculate a 95% confidence interval
    ci_lower = np.percentile(total_claims, 2.5)
    ci_upper = np.percentile(total_claims, 97.5)
    print(f"95% confidence interval: ({ci_lower:.2f}M, {ci_upper:.2f}M)")
    
    # Return the simulated total claims and all individual claims
    return np.array(total_claims), all_individual_claims

def main():
    """Main function to execute the analysis."""
    print("Loading data...")
    data = load_data()
    
    print("Creating histograms...")
    plot_histograms(data)
    
    print("Calculating theoretical values...")
    theory = calculate_theoretical_values(data)
    
    # Use the composite simulation method with 97% threshold
    # This provides better tail modeling than pure empirical sampling
    # while maintaining realistic behavior for the bulk of claims
    print("Simulating total claims using composite approach with 97% threshold...")
    total_claims, individual_claims = simulate_claims_composite(data, threshold_percentile=97)
    
    # Create QQ plot for total claims
    print("Creating QQ plot...")
    create_qq_plot(total_claims)
    
    simulation = {
        "expected_value": np.mean(total_claims),
        "variance": np.var(total_claims),
        "std_deviation": np.std(total_claims)
    }
    
    print("Analyzing reinsurance impact...")
    reinsurance_results = analyze_reinsurance_impact(total_claims, individual_claims)
    
    # Summarize results
    summarize_results(theory, simulation, reinsurance_results)
    
    # Create time series visualization of historical claims with breakdown by claim size
    print("Creating historical year comparison with claim size breakdown...")
    plt.figure(figsize=FIGSIZE)
    df = pd.read_csv("danishDataFinal.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    
    # Add size categories
    df["Size"] = pd.cut(
        df["Loss"], 
        bins=[0, 5, 20, float('inf')], 
        labels=["0-5M", "5-20M", ">20M"]
    )
    
    # Calculate claim totals by year and size category
    yearly_by_size = df.groupby(["Year", "Size"])["Loss"].sum().unstack().fillna(0)
    
    # Create stacked bar chart
    yearly_by_size.plot(kind="bar", stacked=True, ax=plt.gca(), 
                      colormap="viridis", width=0.8)
    
    # Ensure all years are shown on x-axis
    plt.xticks(rotation=45)
    
    plt.title("Historical Total Claims by Year and Size Category")
    plt.xlabel("Year")
    plt.ylabel("Total Claim Amount (in millions)")
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(title="Claim Size")
    
    # Add total values on top of each bar
    yearly_total = yearly_by_size.sum(axis=1)
    for i, total in enumerate(yearly_total):
        plt.text(i, total + 5, f"{total:.1f}M", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("historical_year_comparison.png", dpi=300, bbox_inches="tight")
    print("Historical year comparison plot created and saved.")
    
    print("Analysis complete. All outputs saved.")

if __name__ == "__main__":
    main()