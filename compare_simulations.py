"""
Compare different simulation methods for modeling fire damage claims:

1. Empirical Distribution: Direct resampling from historical data
2. Composite Distribution: Split at 95th percentile threshold
3. Composite Distribution: Split at 99th percentile threshold

Evaluates each method's impact on reinsurance calculations and tail behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from analysis import load_data, apply_xl_reinsurance_to_individual

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
NUM_SIMULATIONS = 10000  # Number of simulations for each method
POISSON_LAMBDA = 567  # Expected number of claims per year
QUANTILES = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]  # Quantiles to compute

# Reinsurance parameters
M1, L1 = 10, 40  # First contract: M = 10M, L = 40M (capped)
M2, L2 = 10, np.inf  # Second contract: M = 10M, L = ∞ (no upper cap)

def simulate_empirical(data, num_simulations=NUM_SIMULATIONS, lambda_=POISSON_LAMBDA):
    """
    Simulate total claim amounts using pure empirical sampling.
    
    Args:
        data: DataFrame containing the claims data
        num_simulations: Number of simulations to run
        lambda_: Expected number of claims per year
        
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

def simulate_composite(data, num_simulations=NUM_SIMULATIONS, lambda_=POISSON_LAMBDA, threshold_percentile=95):
    """
    Simulate total claim amounts using a composite distribution approach:
    - Poisson distribution for claim frequency
    - Separate parametric distributions for non-extreme and extreme claims
    
    Args:
        data: DataFrame containing the claims data
        num_simulations: Number of simulations to run
        lambda_: Expected number of claims per year
        threshold_percentile: Percentile to split between non-extreme and extreme claims
        
    Returns:
        Tuple containing:
        - Array of simulated total claim amounts
        - List of lists, where each inner list contains the individual claims for a simulation
        - Dictionary with fitted distribution information
    """
    claims_data = data["Loss"].values
    
    # Split the data into non-extreme and extreme claims
    threshold = np.percentile(claims_data, threshold_percentile)
    non_extreme = claims_data[claims_data <= threshold]
    extreme = claims_data[claims_data > threshold]
    
    print(f"Splitting claims at {threshold:.2f}M ({threshold_percentile}%)")
    print(f"Non-extreme claims: {len(non_extreme)} ({len(non_extreme)/len(claims_data):.1%})")
    print(f"Extreme claims: {len(extreme)} ({len(extreme)/len(claims_data):.1%})")
    
    # Fit distributions to non-extreme claims
    distributions_non_extreme = [
        stats.lognorm, stats.gamma, stats.weibull_min
    ]
    
    best_non_extreme_dist = None
    best_non_extreme_params = None
    best_non_extreme_sse = np.inf
    
    for dist in distributions_non_extreme:
        try:
            params = dist.fit(non_extreme)
            cdf_fitted = dist.cdf(sorted(non_extreme), *params)
            empirical_cdf = np.arange(1, len(non_extreme) + 1) / len(non_extreme)
            sse = np.sum((cdf_fitted - empirical_cdf) ** 2)
            
            if sse < best_non_extreme_sse:
                best_non_extreme_dist = dist
                best_non_extreme_params = params
                best_non_extreme_sse = sse
        except Exception as e:
            print(f"Error fitting {dist.name} to non-extreme claims: {str(e)}")
    
    # Fit distributions to extreme claims
    distributions_extreme = [
        stats.pareto, stats.genpareto, stats.lognorm
    ]
    
    best_extreme_dist = None
    best_extreme_params = None
    best_extreme_sse = np.inf
    
    for dist in distributions_extreme:
        try:
            params = dist.fit(extreme)
            cdf_fitted = dist.cdf(sorted(extreme), *params)
            empirical_cdf = np.arange(1, len(extreme) + 1) / len(extreme)
            sse = np.sum((cdf_fitted - empirical_cdf) ** 2)
            
            if sse < best_extreme_sse:
                best_extreme_dist = dist
                best_extreme_params = params
                best_extreme_sse = sse
        except Exception as e:
            print(f"Error fitting {dist.name} to extreme claims: {str(e)}")
    
    print(f"Best distribution for non-extreme claims: {best_non_extreme_dist.name}")
    print(f"Best distribution for extreme claims: {best_extreme_dist.name}")
    
    # Store all simulated total claims
    total_claims = []
    all_individual_claims = []
    
    # Probability of an extreme claim
    p_extreme = 1 - threshold_percentile/100
    
    for _ in range(num_simulations):
        # Simulate number of claims using Poisson distribution
        num_claims = np.random.poisson(lambda_)
        
        if num_claims > 0:
            # For each claim, decide if it's extreme or non-extreme
            claim_types = np.random.choice(['non-extreme', 'extreme'], 
                                          size=num_claims, 
                                          p=[1-p_extreme, p_extreme])
            
            # Generate individual claims
            individual_claims = []
            for claim_type in claim_types:
                if claim_type == 'non-extreme':
                    # Sample from fitted non-extreme distribution
                    claim = best_non_extreme_dist.rvs(*best_non_extreme_params, size=1)[0]
                    # Ensure claim is within non-extreme range (numerical issues can cause values outside the range)
                    claim = min(claim, threshold)
                    claim = max(claim, 0)
                else:
                    # Sample from fitted extreme distribution
                    claim = best_extreme_dist.rvs(*best_extreme_params, size=1)[0]
                    # Ensure claim is within extreme range (numerical issues can cause values outside the range)
                    claim = max(claim, threshold)
                
                individual_claims.append(claim)
            
            total_claim = np.sum(individual_claims)
            all_individual_claims.append(np.array(individual_claims))
        else:
            total_claim = 0
            all_individual_claims.append(np.array([]))
        
        total_claims.append(total_claim)
    
    fit_info = {
        "threshold": threshold,
        "non_extreme_dist": best_non_extreme_dist.name,
        "non_extreme_params": best_non_extreme_params,
        "extreme_dist": best_extreme_dist.name,
        "extreme_params": best_extreme_params
    }
    
    return np.array(total_claims), all_individual_claims, fit_info

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
    
    return {
        "no_reinsurance": no_reinsurance_stats,
        "reinsurance_capped": reinsured_stats1,
        "reinsurance_uncapped": reinsured_stats2
    }

def compare_simulation_methods():
    """
    Compare different simulation methods and their impact on reinsurance analysis.
    """
    print("Loading data...")
    data = load_data()
    
    # Run empirical simulation
    print("\nRunning empirical simulation...")
    start_time = time.time()
    empirical_total, empirical_individual = simulate_empirical(data)
    empirical_time = time.time() - start_time
    print(f"Empirical simulation completed in {empirical_time:.2f} seconds")
    
    # Run composite simulation with 95% threshold
    print("\nRunning composite simulation with 95% threshold...")
    start_time = time.time()
    composite95_total, composite95_individual, fit_info95 = simulate_composite(data, threshold_percentile=95)
    composite95_time = time.time() - start_time
    print(f"Composite simulation (95%) completed in {composite95_time:.2f} seconds")
    
    # Run composite simulation with 99% threshold
    print("\nRunning composite simulation with 99% threshold...")
    start_time = time.time()
    composite99_total, composite99_individual, fit_info99 = simulate_composite(data, threshold_percentile=99)
    composite99_time = time.time() - start_time
    print(f"Composite simulation (99%) completed in {composite99_time:.2f} seconds")
    
    # Analyze reinsurance impact for each method
    print("\nAnalyzing reinsurance impact...")
    reinsurance_empirical = analyze_reinsurance_impact(empirical_total, empirical_individual)
    reinsurance_composite95 = analyze_reinsurance_impact(composite95_total, composite95_individual)
    reinsurance_composite99 = analyze_reinsurance_impact(composite99_total, composite99_individual)
    
    # Create summary statistics comparison
    print("\nSummary Statistics Comparison:")
    summary_stats = {
        "Method": ["Empirical", "Composite (95%)", "Composite (99%)"],
        "Mean": [
            np.mean(empirical_total),
            np.mean(composite95_total),
            np.mean(composite99_total)
        ],
        "Std": [
            np.std(empirical_total),
            np.std(composite95_total),
            np.std(composite99_total)
        ],
        "Min": [
            np.min(empirical_total),
            np.min(composite95_total),
            np.min(composite99_total)
        ],
        "Max": [
            np.max(empirical_total),
            np.max(composite95_total),
            np.max(composite99_total)
        ],
        "50%": [
            np.percentile(empirical_total, 50),
            np.percentile(composite95_total, 50),
            np.percentile(composite99_total, 50)
        ],
        "95%": [
            np.percentile(empirical_total, 95),
            np.percentile(composite95_total, 95),
            np.percentile(composite99_total, 95)
        ],
        "99%": [
            np.percentile(empirical_total, 99),
            np.percentile(composite95_total, 99),
            np.percentile(composite99_total, 99)
        ],
        "99.9%": [
            np.percentile(empirical_total, 99.9),
            np.percentile(composite95_total, 99.9),
            np.percentile(composite99_total, 99.9)
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("simulation_methods_comparison.csv", index=False)
    
    # Create reinsurance impact comparison
    print("\nReinsurance Impact Comparison:")
    reinsurance_stats = {
        "Method": ["Empirical", "Composite (95%)", "Composite (99%)"],
        "Mean (No Reinsurance)": [
            reinsurance_empirical["no_reinsurance"]["expected_value"],
            reinsurance_composite95["no_reinsurance"]["expected_value"],
            reinsurance_composite99["no_reinsurance"]["expected_value"]
        ],
        "Std (No Reinsurance)": [
            reinsurance_empirical["no_reinsurance"]["std_deviation"],
            reinsurance_composite95["no_reinsurance"]["std_deviation"],
            reinsurance_composite99["no_reinsurance"]["std_deviation"]
        ],
        "Mean (M=10, L=40)": [
            reinsurance_empirical["reinsurance_capped"]["expected_value"],
            reinsurance_composite95["reinsurance_capped"]["expected_value"],
            reinsurance_composite99["reinsurance_capped"]["expected_value"]
        ],
        "Std (M=10, L=40)": [
            reinsurance_empirical["reinsurance_capped"]["std_deviation"],
            reinsurance_composite95["reinsurance_capped"]["std_deviation"],
            reinsurance_composite99["reinsurance_capped"]["std_deviation"]
        ],
        "Mean (M=10, L=∞)": [
            reinsurance_empirical["reinsurance_uncapped"]["expected_value"],
            reinsurance_composite95["reinsurance_uncapped"]["expected_value"],
            reinsurance_composite99["reinsurance_uncapped"]["expected_value"]
        ],
        "Std (M=10, L=∞)": [
            reinsurance_empirical["reinsurance_uncapped"]["std_deviation"],
            reinsurance_composite95["reinsurance_uncapped"]["std_deviation"],
            reinsurance_composite99["reinsurance_uncapped"]["std_deviation"]
        ]
    }
    
    reinsurance_df = pd.DataFrame(reinsurance_stats)
    print(reinsurance_df.to_string(index=False))
    reinsurance_df.to_csv("reinsurance_impact_comparison.csv", index=False)
    
    # Create visualization of the simulation method comparisons
    plt.figure(figsize=(12, 6))
    sns.kdeplot(empirical_total, label="Empirical", alpha=0.7)
    sns.kdeplot(composite95_total, label="Composite (95%)", alpha=0.7)
    sns.kdeplot(composite99_total, label="Composite (99%)", alpha=0.7)
    plt.title("Distribution Comparison of Simulation Methods")
    plt.xlabel("Total Claim Amount (in millions)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("simulation_methods_comparison.png", dpi=300, bbox_inches="tight")
    
    # Create tail comparison plot (focusing on high quantiles)
    plt.figure(figsize=(12, 6))
    quantiles = np.arange(90, 100, 0.1)
    empirical_quantiles = np.percentile(empirical_total, quantiles)
    composite95_quantiles = np.percentile(composite95_total, quantiles)
    composite99_quantiles = np.percentile(composite99_total, quantiles)
    
    plt.plot(quantiles, empirical_quantiles, label="Empirical")
    plt.plot(quantiles, composite95_quantiles, label="Composite (95%)")
    plt.plot(quantiles, composite99_quantiles, label="Composite (99%)")
    plt.title("Tail Behavior Comparison (90th to 99.9th percentile)")
    plt.xlabel("Percentile")
    plt.ylabel("Total Claim Amount (in millions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("simulation_tail_comparison.png", dpi=300, bbox_inches="tight")
    
    # Return results for potential further analysis
    return {
        "empirical": {
            "total": empirical_total,
            "individual": empirical_individual,
            "reinsurance": reinsurance_empirical
        },
        "composite95": {
            "total": composite95_total,
            "individual": composite95_individual,
            "reinsurance": reinsurance_composite95,
            "fit_info": fit_info95
        },
        "composite99": {
            "total": composite99_total,
            "individual": composite99_individual,
            "reinsurance": reinsurance_composite99,
            "fit_info": fit_info99
        }
    }

if __name__ == "__main__":
    results = compare_simulation_methods()
    
    print("\nAnalysis completed. Results saved to CSV files and visualizations created.") 