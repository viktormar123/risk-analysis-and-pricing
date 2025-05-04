"""
Compare different methods for simulating insurance claims using a single train-test split:

1. Purely Empirical: Direct resampling from historical data
2. Hybrid Methods:
   - 97% empirical + 3% theoretical
   - 97.5% empirical + 2.5% theoretical
3. Purely Theoretical: Fitting one parametric distribution to all claims

Uses a single train-test split (90% train, 10% test) with stratified sampling to ensure
similar distributions. Includes comprehensive metrics for distribution comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split

# Ensure ClaimDensityPlots directory exists
os.makedirs("ClaimDensityPlots", exist_ok=True)

# Set different random seed for train-test split only
SPLIT_SEED = 456  # Changed seed for train-test split
SIMULATION_SEED = 42

# Parameters
NUM_SIMULATIONS = 10000  # Increased for more stable results
POISSON_LAMBDA = 567  # Expected number of claims per year
EXTREME_THRESHOLD = 10  # Base threshold for extreme claims (in millions)
# Specific thresholds to test based on previous findings
THRESHOLD_PERCENTILES = [97.0, 97.5]
# Test set size (percentage)
TEST_SIZE = 0.50

def load_data(file_path="danishDataFinal.csv"):
    """Load the Danish fire damage data."""
    data = pd.read_csv(file_path)
    # Convert Loss to numeric if it's not already
    data["Loss"] = pd.to_numeric(data["Loss"])
    return data

def create_stratified_split(claims, test_size=TEST_SIZE, extreme_threshold=EXTREME_THRESHOLD):
    """
    Create a stratified train-test split ensuring similar distribution of 
    extreme and non-extreme claims using scikit-learn.
    
    Args:
        claims: Array of claim amounts
        test_size: Proportion of data to use for testing
        extreme_threshold: Threshold for extreme claims
        
    Returns:
        Tuple containing (train_indices, test_indices)
    """
    # Set random state for reproducible train-test split
    np.random.seed(SPLIT_SEED)
    
    # Create binary labels for stratification (1 for extreme claims, 0 for non-extreme)
    is_extreme = (claims > extreme_threshold).astype(int)
    
    # Get indices array
    indices = np.arange(len(claims))
    
    # Use scikit-learn's train_test_split with stratification
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=SPLIT_SEED, 
        stratify=is_extreme
    )
    
    # Reset seed for simulations
    #np.random.seed(SIMULATION_SEED)
    
    # Verify stratification
    train_claims = claims[train_indices]
    test_claims = claims[test_indices]
    
    train_extreme_pct = np.mean(train_claims > extreme_threshold) * 100
    test_extreme_pct = np.mean(test_claims > extreme_threshold) * 100
    
    print(f"Train set: {len(train_indices)} claims, {train_extreme_pct:.1f}% extreme")
    print(f"Test set: {len(test_indices)} claims, {test_extreme_pct:.1f}% extreme")
    
    return train_indices, test_indices

def create_random_split(claims, test_size=TEST_SIZE):
    """
    Create a simple random train-test split without stratification.
    
    Args:
        claims: Array of claim amounts
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple containing (train_indices, test_indices)
    """
    # Set random state for reproducible train-test split
    np.random.seed(SPLIT_SEED)
    
    # Get indices array
    indices = np.arange(len(claims))
    
    # Use scikit-learn's train_test_split without stratification
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=SPLIT_SEED
    )
    
    # Verify distribution of extreme claims
    train_claims = claims[train_indices]
    test_claims = claims[test_indices]
    
    train_extreme_pct = np.mean(train_claims > EXTREME_THRESHOLD) * 100
    test_extreme_pct = np.mean(test_claims > EXTREME_THRESHOLD) * 100
    
    print(f"Train set: {len(train_indices)} claims, {train_extreme_pct:.1f}% extreme")
    print(f"Test set: {len(test_indices)} claims, {test_extreme_pct:.1f}% extreme")
    
    return train_indices, test_indices

def simulate_empirical(train_claims, num_claims, num_simulations=NUM_SIMULATIONS):
    """
    Simulate claims using pure empirical sampling.
    
    Args:
        train_claims: Training claims data to sample from
        num_claims: Number of claims to simulate in each run
        num_simulations: Number of simulations to run
        
    Returns:
        Simulated total claims for each simulation
    """
    total_claims = []
    
    for _ in range(num_simulations):
        # Sample claims from empirical distribution
        if num_claims > 0:
            sampled_claims = np.random.choice(train_claims, size=num_claims, replace=True)
            total_claim = np.sum(sampled_claims)
        else:
            total_claim = 0
        
        total_claims.append(total_claim)
    
    return np.array(total_claims)

def plot_distribution_pdf(dist, params, data, name, color, save_path):
    """
    Plot the PDF of a fitted distribution against a histogram of the data.
    
    Args:
        dist: The scipy.stats distribution object
        params: Parameters for the distribution
        data: The data used for fitting
        name: Name of the distribution for labeling
        color: Color for the distribution line
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot histogram of data with density=True for PDF comparison
    hist, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.6, color='grey', label='Data Histogram')
    
    # Get x values for plotting the PDF
    x = np.linspace(min(data), max(data) * 1.1, 1000)
    
    # Calculate and plot the PDF
    pdf = dist.pdf(x, *params)
    plt.plot(x, pdf, color=color, lw=2, label=f'{name} PDF')
    
    # Add labels and title
    plt.xlabel('Claim Amount (in millions)')
    plt.ylabel('Probability Density')
    plt.title(f'PDF of Fitted {name} Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_compare_distributions(dist_infos, claims, threshold_percentiles, save_path):
    """
    Plot and compare PDFs of all fitted distributions.
    
    Args:
        dist_infos: Dictionary containing information about fitted distributions
        claims: All claim data
        threshold_percentiles: List of threshold percentiles for hybrid models
        save_path: Path to save the comparison plot
    """
    plt.figure(figsize=(14, 10))
    
    # Define a range of claim values for plotting
    x = np.linspace(0, np.percentile(claims, 99.5) * 1.2, 1000)
    
    # Plot histogram of all claims with density=True for PDF comparison
    hist, bins, _ = plt.hist(claims, bins=50, density=True, alpha=0.3, color='grey', label='All Claims Histogram')
    
    # Plot theoretical distribution if available
    if 'theoretical' in dist_infos and 'best_dist_name' in dist_infos['theoretical']:
        theo_info = dist_infos['theoretical']
        dist_name = theo_info['best_dist_name']
        params = theo_info['best_dist_params']
        dist = getattr(stats, dist_name)
        
        try:
            pdf = dist.pdf(x, *params)
            plt.plot(x, pdf, lw=2, label=f'Theoretical ({dist_name})')
        except Exception as e:
            print(f"Error plotting theoretical PDF: {e}")
    
    # Plot hybrid distributions
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, percentile in enumerate(threshold_percentiles):
        model_name = f"hybrid_{percentile}"
        
        if model_name in dist_infos and 'best_dist_name' in dist_infos[model_name]:
            hybrid_info = dist_infos[model_name]
            
            # Get the threshold
            threshold = hybrid_info['threshold']
            
            # Plot vertical line at threshold
            plt.axvline(threshold, color=colors[i % len(colors)], linestyle='--', alpha=0.5, 
                        label=f'Threshold {percentile}%')
            
            # Plot the fitted distribution for extreme values
            dist_name = hybrid_info['best_dist_name']
            params = hybrid_info['best_dist_params']
            dist = getattr(stats, dist_name)
            
            # For hybrid models, we only apply the distribution to values above threshold
            # and we need to scale by the probability of exceeding the threshold
            p_extreme = hybrid_info['p_extreme']
            
            try:
                # We need to plot this differently - only for x values above threshold
                x_extreme = x[x >= threshold]
                if len(x_extreme) > 0:
                    pdf_extreme = dist.pdf(x_extreme, *params) * p_extreme
                    plt.plot(x_extreme, pdf_extreme, color=colors[i % len(colors)], lw=2, 
                             label=f'Hybrid {percentile}% Tail ({dist_name})')
            except Exception as e:
                print(f"Error plotting hybrid {percentile}% PDF: {e}")
    
    # Set logarithmic scale for y-axis to better visualize the tails
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Claim Amount (in millions)')
    plt.ylabel('Probability Density (log scale)')
    plt.title('Comparison of Fitted Distribution PDFs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def simulate_hybrid(train_claims, num_claims, threshold_percentile, num_simulations=NUM_SIMULATIONS):
    """
    Simulate claims using a hybrid approach - empirical for bulk, theoretical for tail.
    
    Args:
        train_claims: Training claims data
        num_claims: Number of claims to simulate in each run
        threshold_percentile: Percentile to split between empirical and theoretical
        num_simulations: Number of simulations to run
        
    Returns:
        Tuple containing: (simulated total claims, distribution info dict)
    """
    threshold = np.percentile(train_claims, threshold_percentile)
    non_extreme = train_claims[train_claims <= threshold]
    extreme = train_claims[train_claims > threshold]
    
    # Store info on the distribution and fitting
    dist_info = {
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "non_extreme_count": len(non_extreme),
        "extreme_count": len(extreme),
        "extreme_ratio": len(extreme)/len(train_claims),
    }
    
    print(f"  Hybrid {threshold_percentile}%: Split at {threshold:.2f}M, {len(extreme)} extreme claims ({len(extreme)/len(train_claims):.1%})")
    
    # Fit distribution to extreme claims
    distributions = [stats.pareto, stats.genpareto, stats.lognorm]
    best_dist = None
    best_params = None
    best_sse = np.inf
    
    for dist in distributions:
        try:
            params = dist.fit(extreme)
            # Calculate theoretical moments for the best distribution
            cdf_fitted = dist.cdf(sorted(extreme), *params)
            empirical_cdf = np.arange(1, len(extreme) + 1) / len(extreme)
            sse = np.sum((cdf_fitted - empirical_cdf) ** 2)
            
            if sse < best_sse:
                best_dist = dist
                best_params = params
                best_sse = sse
        except Exception as e:
            print(f"    Error fitting {dist.name}: {str(e)}")
            continue
    
    if best_dist is not None:
        print(f"    Best distribution for extreme claims: {best_dist.name}")
        # Store distribution info
        dist_info["best_dist_name"] = best_dist.name
        dist_info["best_dist_params"] = best_params
        
        # Plot the fitted distribution for extreme claims
        plot_distribution_pdf(
            best_dist, 
            best_params, 
            extreme,
            f"Hybrid {threshold_percentile}% Extreme ({best_dist.name})",
            "red",
            f"ClaimDensityPlots/hybrid_{threshold_percentile}_extreme_pdf.png"
        )
        
        # Add theoretical moments if possible
        try:
            if best_dist.name == "genpareto":
                # Parameters: shape, loc, scale
                c, loc, scale = best_params
                if c < 1:  # Mean exists only if shape parameter < 1
                    dist_info["extreme_theoretical_mean"] = loc + scale/(1-c)
                else:
                    dist_info["extreme_theoretical_mean"] = np.inf
                
                if c < 0.5:  # Variance exists only if shape parameter < 0.5
                    dist_info["extreme_theoretical_var"] = scale**2/((1-c)**2 * (1-2*c))
                else:
                    dist_info["extreme_theoretical_var"] = np.inf
            elif best_dist.name == "pareto":
                # Parameters for scipy.stats.pareto: shape, loc, scale
                b, loc, scale = best_params
                if b > 1:  # Mean exists only if shape parameter > 1
                    dist_info["extreme_theoretical_mean"] = loc + scale * b / (b - 1)
                else:
                    dist_info["extreme_theoretical_mean"] = np.inf
                
                if b > 2:  # Variance exists only if shape parameter > 2
                    dist_info["extreme_theoretical_var"] = scale**2 * b / ((b - 1)**2 * (b - 2))
                else:
                    dist_info["extreme_theoretical_var"] = np.inf
            elif best_dist.name == "lognorm":
                # Parameters: shape, loc, scale
                s, loc, scale = best_params
                dist_info["extreme_theoretical_mean"] = loc + scale * np.exp(s**2/2)
                dist_info["extreme_theoretical_var"] = scale**2 * np.exp(s**2) * (np.exp(s**2) - 1)
        except Exception as e:
            print(f"    Error calculating theoretical moments: {str(e)}")
    else:
        print("    Warning: Could not fit any distribution to extreme claims, using empirical sampling instead")
        return simulate_empirical(train_claims, num_claims, num_simulations), {"error": "No distribution fit"}
    
    # Calculate empirical moments for extreme claims
    dist_info["extreme_empirical_mean"] = np.mean(extreme)
    dist_info["extreme_empirical_var"] = np.var(extreme)
    
    # Probability of extreme claim
    p_extreme = 1 - threshold_percentile/100
    dist_info["p_extreme"] = p_extreme
    
    # Simulate claims
    total_claims = []
    for _ in range(num_simulations):
        if num_claims > 0:
            # For each claim, decide if it's extreme or non-extreme
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
                    # Sample from fitted theoretical distribution for extreme claims
                    claim = best_dist.rvs(*best_params, size=1)[0]
                    # Ensure claim is at least at the threshold
                    claim = max(claim, threshold)
                
                individual_claims.append(claim)
            
            total_claim = np.sum(individual_claims)
        else:
            total_claim = 0
            
        total_claims.append(total_claim)
    
    return np.array(total_claims), dist_info

def simulate_theoretical(train_claims, num_claims, num_simulations=NUM_SIMULATIONS):
    """
    Simulate claims using purely theoretical distributions fitted to all claims.
    
    Args:
        train_claims: Training claims data
        num_claims: Number of claims to simulate in each run
        num_simulations: Number of simulations to run
        
    Returns:
        Tuple containing: (simulated total claims, distribution info dict)
    """
    # Fit distributions to all claims
    distributions = [stats.lognorm, stats.gamma, stats.weibull_min, stats.genpareto]
    
    # Store info on the distribution and fitting
    dist_info = {}
    
    best_dist = None
    best_params = None
    best_aic = np.inf
    
    for dist in distributions:
        try:
            params = dist.fit(train_claims)
            log_likelihood = np.sum(dist.logpdf(train_claims, *params))
            k = len(params)
            n = len(train_claims)
            aic = 2 * k - 2 * log_likelihood
            
            if aic < best_aic:
                best_dist = dist
                best_params = params
                best_aic = aic
        except Exception as e:
            print(f"    Error fitting {dist.name}: {str(e)}")
            continue
    
    if best_dist is None:
        print("    Warning: Could not fit any distribution, using empirical sampling instead")
        return simulate_empirical(train_claims, num_claims, num_simulations), {"error": "No distribution fit"}
        
    print(f"    Best overall distribution: {best_dist.name}")
    
    # Store distribution info
    dist_info["best_dist_name"] = best_dist.name
    dist_info["best_dist_params"] = best_params
    
    # Plot the fitted distribution
    plot_distribution_pdf(
        best_dist, 
        best_params, 
        train_claims,
        f"Theoretical ({best_dist.name})",
        "blue",
        f"ClaimDensityPlots/theoretical_pdf.png"
    )
    
    # Add theoretical moments if possible
    try:
        if best_dist.name == "genpareto":
            # Parameters: shape, loc, scale
            c, loc, scale = best_params
            if c < 1:  # Mean exists only if shape parameter < 1
                dist_info["theoretical_mean"] = loc + scale/(1-c)
            else:
                dist_info["theoretical_mean"] = np.inf
            
            if c < 0.5:  # Variance exists only if shape parameter < 0.5
                dist_info["theoretical_var"] = scale**2/((1-c)**2 * (1-2*c))
            else:
                dist_info["theoretical_var"] = np.inf
        elif best_dist.name == "lognorm":
            # Parameters: shape, loc, scale
            s, loc, scale = best_params
            dist_info["theoretical_mean"] = loc + scale * np.exp(s**2/2)
            dist_info["theoretical_var"] = scale**2 * np.exp(s**2) * (np.exp(s**2) - 1)
        elif best_dist.name == "gamma":
            # Parameters: shape, loc, scale
            a, loc, scale = best_params
            dist_info["theoretical_mean"] = loc + a * scale
            dist_info["theoretical_var"] = a * scale**2
        elif best_dist.name == "weibull_min":
            # Parameters: shape, loc, scale
            c, loc, scale = best_params
            from scipy.special import gamma as gamma_func
            dist_info["theoretical_mean"] = loc + scale * gamma_func(1 + 1/c)
            dist_info["theoretical_var"] = scale**2 * (gamma_func(1 + 2/c) - gamma_func(1 + 1/c)**2)
    except Exception as e:
        print(f"    Error calculating theoretical moments: {str(e)}")
    
    # Simulate claims
    total_claims = []
    for _ in range(num_simulations):
        if num_claims > 0:
            # Sample from fitted distribution
            sampled_claims = best_dist.rvs(*best_params, size=num_claims)
            # Ensure all claims are positive
            sampled_claims = np.maximum(sampled_claims, 0)
            total_claim = np.sum(sampled_claims)
        else:
            total_claim = 0
            
        total_claims.append(total_claim)
    
    return np.array(total_claims), dist_info

def calculate_kl_divergence(p_samples, q_samples, bins=20, smoothing=1e-10):
    """
    Calculate the KL divergence between two sets of samples.
    Uses adaptive binning and smoothing to avoid numerical issues.
    
    Args:
        p_samples: First set of samples (reference)
        q_samples: Second set of samples (comparison)
        bins: Number of bins for histogram
        smoothing: Small value to avoid log(0)
        
    Returns:
        KL divergence value
    """
    # Determine bin edges based on combined range
    min_val = min(np.min(p_samples), np.min(q_samples))
    max_val = max(np.max(p_samples), np.max(q_samples)) * 1.1  # Add 10% margin
    
    # Create bins
    bin_edges = np.linspace(min_val, max_val, bins+1)
    
    # Calculate histograms
    p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=True)
    
    # Apply smoothing
    p_hist = p_hist + smoothing
    q_hist = q_hist + smoothing
    
    # Normalize after smoothing
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)
    
    # Calculate KL divergence: KL(p||q)
    return np.sum(p_hist * np.log(p_hist / q_hist))

def evaluate_distributions(predicted_totals, actual_totals, actual_claims=None, name="Model"):
    """
    Evaluate distribution similarity and calculate comprehensive metrics.
    
    Args:
        predicted_totals: Array of predicted total claims
        actual_totals: Array of actual total claims
        actual_claims: Original claim amounts (for additional metrics)
        name: Name of the model for reporting
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic distribution statistics
    metrics["mean"] = np.mean(predicted_totals)
    metrics["actual_mean"] = np.mean(actual_totals)
    metrics["mean_error"] = metrics["mean"] - metrics["actual_mean"]
    metrics["mean_abs_error"] = np.abs(metrics["mean_error"])
    metrics["mean_rel_error"] = metrics["mean_error"] / metrics["actual_mean"] if metrics["actual_mean"] != 0 else np.nan
    
    metrics["std"] = np.std(predicted_totals)
    metrics["actual_std"] = np.std(actual_totals)
    metrics["std_error"] = metrics["std"] - metrics["actual_std"]
    metrics["std_rel_error"] = metrics["std_error"] / metrics["actual_std"] if metrics["actual_std"] != 0 else np.nan
    
    metrics["skewness"] = stats.skew(predicted_totals)
    metrics["actual_skewness"] = stats.skew(actual_totals)
    
    metrics["kurtosis"] = stats.kurtosis(predicted_totals)
    metrics["actual_kurtosis"] = stats.kurtosis(actual_totals)
    
    # Quantile metrics
    quantiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    for q in quantiles:
        pred_q = np.percentile(predicted_totals, q)
        actual_q = np.percentile(actual_totals, q)
        
        metrics[f"q{q}"] = pred_q
        metrics[f"actual_q{q}"] = actual_q
        metrics[f"q{q}_error"] = pred_q - actual_q
        metrics[f"q{q}_rel_error"] = (pred_q - actual_q) / actual_q if actual_q != 0 else np.nan
    
    # Tail metrics (95th percentile and above)
    tail_threshold = np.percentile(predicted_totals, 95)
    actual_tail_threshold = np.percentile(actual_totals, 95)
    
    tail_predicted = predicted_totals[predicted_totals >= tail_threshold]
    actual_tail = actual_totals[actual_totals >= actual_tail_threshold]
    
    if len(tail_predicted) > 0 and len(actual_tail) > 0:
        metrics["tail_mean"] = np.mean(tail_predicted)
        metrics["actual_tail_mean"] = np.mean(actual_tail)
        metrics["tail_std"] = np.std(tail_predicted)
        metrics["actual_tail_std"] = np.std(actual_tail)
        
        # Calculate KL divergence for the full distribution
        try:
            metrics["kl_full"] = calculate_kl_divergence(actual_totals, predicted_totals)
        except Exception as e:
            print(f"    Warning: KL divergence calculation failed for full distribution: {str(e)}")
            metrics["kl_full"] = np.nan
        
        # Calculate KL divergence for the tail
        try:
            metrics["kl_tail"] = calculate_kl_divergence(actual_tail, tail_predicted)
        except Exception as e:
            print(f"    Warning: KL divergence calculation failed for tail: {str(e)}")
            metrics["kl_tail"] = np.nan
    
    # Statistical tests
    try:
        ks_stat, ks_pvalue = stats.ks_2samp(predicted_totals, actual_totals)
        metrics["ks_stat"] = ks_stat
        metrics["ks_pvalue"] = ks_pvalue
    except Exception as e:
        print(f"    Warning: KS test failed: {str(e)}")
        metrics["ks_stat"] = metrics["ks_pvalue"] = np.nan
    
    try:
        anderson_result = stats.anderson_ksamp([predicted_totals, actual_totals])
        metrics["anderson_stat"] = anderson_result.statistic
        metrics["anderson_critical_values"] = anderson_result.critical_values[2]  # 5% significance
        metrics["anderson_significance_level"] = 0.05  # 5% significance
    except Exception as e:
        print(f"    Warning: Anderson-Darling test failed: {str(e)}")
        metrics["anderson_stat"] = metrics["anderson_critical_values"] = metrics["anderson_significance_level"] = np.nan
    
    # Wasserstein distance (Earth Mover's Distance)
    try:
        metrics["wasserstein"] = stats.wasserstein_distance(predicted_totals, actual_totals)
    except Exception as e:
        print(f"    Warning: Wasserstein distance calculation failed: {str(e)}")
        metrics["wasserstein"] = np.nan
    
    # Print summary
    print(f"  {name} Summary:")
    print(f"    Mean: {metrics['mean']:.2f} (Actual: {metrics['actual_mean']:.2f}, Error: {metrics['mean_error']:.2f}, Rel: {metrics['mean_rel_error']*100:.2f}%)")
    print(f"    Std: {metrics['std']:.2f} (Actual: {metrics['actual_std']:.2f}, Error: {metrics['std_error']:.2f}, Rel: {metrics['std_rel_error']*100:.2f}%)")
    print(f"    Q99: {metrics['q99']:.2f} (Actual: {metrics['actual_q99']:.2f}, Error: {metrics['q99_error']:.2f}, Rel: {metrics['q99_rel_error']*100:.2f}%)")
    print(f"    Tail Mean: {metrics['tail_mean']:.2f} (Actual: {metrics['actual_tail_mean']:.2f})")
    print(f"    KL Divergence: Full={metrics.get('kl_full', 'N/A')}, Tail={metrics.get('kl_tail', 'N/A')}")
    print(f"    KS Test: Stat={metrics['ks_stat']:.4f}, p-value={metrics['ks_pvalue']:.4f}")
    print(f"    Wasserstein Distance: {metrics['wasserstein']:.4f}")
    
    return metrics

def compare_models(run_seed=SPLIT_SEED, use_stratified=True):
    """
    Compare different claim simulation models using a single train-test split.
    
    Args:
        run_seed: Random seed for this run
        use_stratified: Whether to use stratified split or random split
        
    Returns:
        Tuple of (results, dist_infos) dictionaries
    """
    # Set random seed for this run
    SPLIT_SEED = run_seed
    np.random.seed(run_seed)
    
    # Load data
    print(f"Run with seed {run_seed} - {'Stratified' if use_stratified else 'Random'} split")
    print("Loading data...")
    data = load_data()
    claims = data["Loss"].values
    
    # Create train-test split
    print("Creating train-test split...")
    if use_stratified:
        train_idx, test_idx = create_stratified_split(claims)
    else:
        train_idx, test_idx = create_random_split(claims)
    
    train_claims = claims[train_idx]
    test_claims = claims[test_idx]
    
    # Number of claims to simulate for each run (use test set size)
    num_claims_test = len(test_idx)
    
    # Plot training and test set distributions for comparison
    print("Plotting train vs test distribution comparison...")
    
    # Regular scale histogram comparison
    plt.figure(figsize=(12, 8))
    plt.hist(train_claims, bins=50, density=True, alpha=0.5, color='blue', label='Training Set')
    plt.hist(test_claims, bins=50, density=True, alpha=0.5, color='red', label='Test Set')
    plt.axvline(EXTREME_THRESHOLD, color='black', linestyle='--', label=f'Extreme Threshold ({EXTREME_THRESHOLD}M)')
    plt.xlabel('Claim Amount (in millions)')
    plt.ylabel('Probability Density')
    plot_title = f'Comparison of Train-Test Distributions ({"Stratified" if use_stratified else "Random"} Split, Seed={run_seed})'
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, np.percentile(np.concatenate([train_claims, test_claims]), 99.5))
    plt.savefig(f"ClaimDensityPlots/train_test_comparison_seed{run_seed}_{'strat' if use_stratified else 'rand'}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bootstrap samples of the test set for reference distribution
    print("Creating bootstrap samples of test set for reference...")
    # Ensure consistent random state for bootstrap
    np.random.seed(SIMULATION_SEED)
    bootstrap_size = 10000
    bootstrap_samples = []
    for _ in range(bootstrap_size):
        sampled_claims = np.random.choice(test_claims, size=len(test_claims), replace=True)
        bootstrap_samples.append(np.sum(sampled_claims))
    bootstrap_samples = np.array(bootstrap_samples)
    
    # Calculate the actual total and distribution for the test set
    test_total = np.sum(test_claims)
    print(f"Test set total: {test_total:.2f}M")
    print(f"Bootstrap: mean={np.mean(bootstrap_samples):.2f}, std={np.std(bootstrap_samples):.2f}")
    
    # Results storage
    results = {}
    dist_infos = {}
    
    # Simulate with empirical model
    print("\nSimulating with empirical model...")
    # Reset seed before each simulation
    np.random.seed(SIMULATION_SEED)
    start_time = time.time()
    empirical_totals = simulate_empirical(train_claims, num_claims_test)
    empirical_time = time.time() - start_time
    
    # Store simulation results for later analysis
    results["empirical"] = {"totals": empirical_totals}
    
    # Evaluate empirical model
    empirical_metrics = evaluate_distributions(empirical_totals, bootstrap_samples, test_claims, "Empirical")
    empirical_metrics["time"] = empirical_time
    results["empirical"].update(empirical_metrics)
    
    # Simulate with theoretical model
    print("\nSimulating with theoretical model (single distribution)...")
    # Reset seed before each simulation
    np.random.seed(SIMULATION_SEED)
    start_time = time.time()
    theoretical_totals, theoretical_info = simulate_theoretical(train_claims, num_claims_test)
    theoretical_time = time.time() - start_time
    
    # Store simulation results for later analysis
    results["theoretical"] = {"totals": theoretical_totals}
    
    # Evaluate theoretical model
    theoretical_metrics = evaluate_distributions(theoretical_totals, bootstrap_samples, test_claims, "Theoretical")
    theoretical_metrics["time"] = theoretical_time
    results["theoretical"].update(theoretical_metrics)
    dist_infos["theoretical"] = theoretical_info
    
    # Simulate with hybrid models
    for percentile in THRESHOLD_PERCENTILES:
        model_name = f"hybrid_{percentile}"
        print(f"\nSimulating with hybrid model (empirical {percentile}% + theoretical {100-percentile}%)...")
        
        # Reset seed before each simulation
        np.random.seed(SIMULATION_SEED)
        start_time = time.time()
        hybrid_totals, hybrid_details = simulate_hybrid(train_claims, num_claims_test, percentile)
        hybrid_time = time.time() - start_time
        
        # Store simulation results for later analysis
        results[model_name] = {"totals": hybrid_totals}
        
        # Evaluate hybrid model
        hybrid_metrics = evaluate_distributions(hybrid_totals, bootstrap_samples, test_claims, f"Hybrid {percentile}%")
        hybrid_metrics["time"] = hybrid_time
        hybrid_metrics["percentile"] = percentile
        hybrid_metrics["empirical_pct"] = percentile
        hybrid_metrics["theoretical_pct"] = 100 - percentile
        hybrid_metrics["totals"] = hybrid_totals  # Store totals for later visualization
        
        # Save detailed information
        hybrid_info[model_name] = hybrid_details
        hybrid_results[model_name] = hybrid_metrics
    
    # For all plots, set the same seed to ensure consistency
    np.random.seed(SIMULATION_SEED)
    
    # Plot empirical distribution of all claims
    plt.figure(figsize=(12, 8))
    plt.hist(train_claims, bins=100, density=True, alpha=0.6, color='grey', label='All Claims')
    plt.axvline(np.percentile(train_claims, 97.0), color='red', linestyle='--', label='97.0% Threshold')
    plt.axvline(np.percentile(train_claims, 97.5), color='blue', linestyle='--', label='97.5% Threshold')
    plt.xlabel('Claim Amount (in millions)')
    plt.ylabel('Probability Density')
    plt.title('Empirical Distribution of Claims')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, np.percentile(train_claims, 99.5))  # Focus on meaningful range
    plt.savefig("ClaimDensityPlots/empirical_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot log-scale version of empirical distribution
    plt.figure(figsize=(12, 8))
    plt.hist(train_claims, bins=100, density=True, alpha=0.6, color='grey', label='All Claims')
    plt.axvline(np.percentile(train_claims, 97.0), color='red', linestyle='--', label='97.0% Threshold')
    plt.axvline(np.percentile(train_claims, 97.5), color='blue', linestyle='--', label='97.5% Threshold')
    plt.xlabel('Claim Amount (in millions)')
    plt.ylabel('Probability Density (log scale)')
    plt.yscale('log')
    plt.title('Empirical Distribution of Claims (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, np.percentile(train_claims, 99.5))  # Focus on meaningful range
    plt.savefig("ClaimDensityPlots/empirical_distribution_log.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot comparison of all fitted distribution PDFs
    plot_compare_distributions(
        dist_infos, 
        train_claims, 
        THRESHOLD_PERCENTILES,
        "ClaimDensityPlots/distribution_comparison.png"
    )
    
    # Create visualizations
    # Distribution comparison
    plt.figure(figsize=(14, 8))
    
    # Plot kernel density estimates
    sns.kdeplot(bootstrap_samples, label="Test Bootstrap", alpha=0.7, linestyle="--")
    sns.kdeplot(empirical_totals, label="Empirical", alpha=0.7)
    sns.kdeplot(theoretical_totals, label="Theoretical", alpha=0.7)
    
    for percentile in THRESHOLD_PERCENTILES:
        model_name = f"hybrid_{percentile}"
        model_totals = results[model_name].get("totals", [])
        if len(model_totals) > 0:
            sns.kdeplot(model_totals, label=f"Hybrid {percentile}%", alpha=0.7)
    
    plt.axvline(test_total, color='r', linestyle='--', label=f"Actual Test Total: {test_total:.1f}M")
    plt.title("Distribution of Simulated Total Claims")
    plt.xlabel("Total Claim Amount (in millions)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("distribution_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Tail comparison
    plt.figure(figsize=(14, 8))
    
    # Plot tail distributions (95th percentile and above)
    tail_quantiles = np.arange(95, 100, 0.1)
    
    # Function to calculate quantiles safely
    def safe_quantile(data, q):
        try:
            return np.percentile(data, q)
        except:
            return np.nan
    
    # Get quantiles for all distributions
    bootstrap_tail = [safe_quantile(bootstrap_samples, q) for q in tail_quantiles]
    empirical_tail = [safe_quantile(empirical_totals, q) for q in tail_quantiles]
    theoretical_tail = [safe_quantile(theoretical_totals, q) for q in tail_quantiles]
    
    hybrid_tails = {}
    for percentile in THRESHOLD_PERCENTILES:
        model_name = f"hybrid_{percentile}"
        model_totals = results[model_name].get("totals", [])
        if len(model_totals) > 0:
            hybrid_tails[percentile] = [safe_quantile(model_totals, q) for q in tail_quantiles]
    
    # Plot all distributions
    plt.plot(tail_quantiles, bootstrap_tail, label="Test Bootstrap", linewidth=2, linestyle="--")
    plt.plot(tail_quantiles, empirical_tail, label="Empirical", linewidth=2)
    plt.plot(tail_quantiles, theoretical_tail, label="Theoretical", linewidth=2)
    
    for percentile in THRESHOLD_PERCENTILES:
        if percentile in hybrid_tails:
            plt.plot(tail_quantiles, hybrid_tails[percentile], 
                   label=f"Hybrid {percentile}%", 
                   linewidth=2)
    
    plt.title("Tail Behavior Comparison (95th to 99.9th percentile)")
    plt.xlabel("Percentile")
    plt.ylabel("Total Claim Amount (in millions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("tail_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create a comprehensive comparison table with ALL metrics
    print("\n=== COMPREHENSIVE COMPARISON ===")
    metrics_to_show = [
        "mean", "actual_mean", "mean_error", "mean_rel_error", 
        "std", "actual_std", "std_error", "std_rel_error",
        "skewness", "actual_skewness",
        "kurtosis", "actual_kurtosis",
        "q50", "actual_q50", "q50_rel_error",
        "q75", "actual_q75", "q75_rel_error",
        "q90", "actual_q90", "q90_rel_error",
        "q95", "actual_q95", "q95_rel_error",
        "q99", "actual_q99", "q99_rel_error", 
        "q99.5", "actual_q99.5", "q99.5_rel_error",
        "q99.9", "actual_q99.9", "q99.9_rel_error",
        "tail_mean", "actual_tail_mean",
        "tail_std", "actual_tail_std",
        "kl_full", "kl_tail", 
        "ks_stat", "ks_pvalue",
        "anderson_stat", "anderson_critical_values",
        "wasserstein",
        "time"
    ]
    
    # Create a DataFrame for nicely formatted output
    comparison = {}
    models = ["empirical", "theoretical"] + [f"hybrid_{p}" for p in THRESHOLD_PERCENTILES]
    
    for metric in metrics_to_show:
        comparison[metric] = []
        for model in models:
            value = results[model].get(metric, np.nan)
            comparison[metric].append(value)
    
    # Create a DataFrame and print
    df = pd.DataFrame(comparison, index=models)
    
    # Print different sections of metrics for better readability
    print("\n--- BASIC STATISTICS ---")
    basic_metrics = ["mean", "actual_mean", "mean_error", "mean_rel_error", 
                     "std", "actual_std", "std_error", "std_rel_error",
                     "skewness", "actual_skewness", "kurtosis", "actual_kurtosis"]
    print(df[basic_metrics].to_string(float_format=lambda x: f"{x:.6f}"))
    
    print("\n--- QUANTILE METRICS ---")
    quantile_metrics = ["q50", "actual_q50", "q50_rel_error",
                        "q75", "actual_q75", "q75_rel_error",
                        "q90", "actual_q90", "q90_rel_error",
                        "q95", "actual_q95", "q95_rel_error",
                        "q99", "actual_q99", "q99_rel_error", 
                        "q99.5", "actual_q99.5", "q99.5_rel_error",
                        "q99.9", "actual_q99.9", "q99.9_rel_error"]
    print(df[quantile_metrics].to_string(float_format=lambda x: f"{x:.6f}"))
    
    print("\n--- TAIL METRICS ---")
    tail_metrics = ["tail_mean", "actual_tail_mean", "tail_std", "actual_tail_std"]
    print(df[tail_metrics].to_string(float_format=lambda x: f"{x:.6f}"))
    
    print("\n--- DISTRIBUTION COMPARISON METRICS ---")
    dist_metrics = ["kl_full", "kl_tail", "ks_stat", "ks_pvalue", 
                    "anderson_stat", "anderson_critical_values", "wasserstein", "time"]
    print(df[dist_metrics].to_string(float_format=lambda x: f"{x:.6f}"))
    
    # Save to CSV
    df.to_csv("model_comparison_results.csv")
    
    # Print distribution information
    print("\n=== DISTRIBUTION DETAILS ===")
    for model, info in dist_infos.items():
        print(f"\n{model.capitalize()} Distribution:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Add run information to results for easier comparison later
    for model_name in results:
        results[model_name]["seed"] = run_seed
        results[model_name]["stratified"] = use_stratified
    
    return results, dist_infos

def compare_hybrid_percentages(percentiles):
    """
    Compare hybrid models with different percentage thresholds using random splits.
    
    Args:
        percentiles: List of percentile thresholds to test (e.g. [94.0, 95.0, 96.0])
        
    Returns:
        Dictionary with results for each percentage
    """
    # Set seed for reproducibility
    np.random.seed(SIMULATION_SEED)
    
    # Load data
    print("Loading data...")
    data = load_data()
    claims = data["Loss"].values
    
    # Create train-test split (random split only)
    print("Creating random train-test split...")
    SPLIT_SEED = 42  # Fixed seed for comparison
    train_idx, test_idx = create_random_split(claims)
    
    train_claims = claims[train_idx]
    test_claims = claims[test_idx]
    
    # Number of claims to simulate for each run
    num_claims_test = len(test_idx)
    
    # Plot training and test set distributions
    print("Plotting train vs test distribution comparison...")
    plt.figure(figsize=(12, 8))
    plt.hist(train_claims, bins=50, density=True, alpha=0.5, color='blue', label='Training Set')
    plt.hist(test_claims, bins=50, density=True, alpha=0.5, color='red', label='Test Set')
    plt.axvline(EXTREME_THRESHOLD, color='black', linestyle='--', label=f'Extreme Threshold ({EXTREME_THRESHOLD}M)')
    plt.xlabel('Claim Amount (in millions)')
    plt.ylabel('Probability Density')
    plt.title('Comparison of Train-Test Distributions (Random Split)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, np.percentile(np.concatenate([train_claims, test_claims]), 99.5))
    plt.savefig("ClaimDensityPlots/hybrid_percentile_train_test_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bootstrap samples of the test set for reference distribution
    print("Creating bootstrap samples of test set for reference...")
    np.random.seed(SIMULATION_SEED)
    bootstrap_size = 10000
    bootstrap_samples = []
    for _ in range(bootstrap_size):
        sampled_claims = np.random.choice(test_claims, size=len(test_claims), replace=True)
        bootstrap_samples.append(np.sum(sampled_claims))
    bootstrap_samples = np.array(bootstrap_samples)
    
    # Calculate the actual total and distribution for the test set
    test_total = np.sum(test_claims)
    print(f"Test set total: {test_total:.2f}M")
    print(f"Bootstrap: mean={np.mean(bootstrap_samples):.2f}, std={np.std(bootstrap_samples):.2f}")
    
    # Storage for results
    results = {}
    hybrid_results = {}
    
    # Simulate with empirical model first as baseline
    print("\nSimulating with empirical model (baseline)...")
    np.random.seed(SIMULATION_SEED)
    start_time = time.time()
    empirical_totals = simulate_empirical(train_claims, num_claims_test)
    empirical_time = time.time() - start_time
    
    # Evaluate empirical model
    empirical_metrics = evaluate_distributions(empirical_totals, bootstrap_samples, test_claims, "Empirical")
    empirical_metrics["time"] = empirical_time
    results["empirical"] = empirical_metrics
    
    # Run hybrid models with different percentiles
    hybrid_info = {}
    for percentile in percentiles:
        model_name = f"hybrid_{percentile}"
        print(f"\nSimulating with hybrid model (empirical {percentile}% + theoretical {100-percentile}%)...")
        
        np.random.seed(SIMULATION_SEED)
        start_time = time.time()
        hybrid_totals, hybrid_details = simulate_hybrid(train_claims, num_claims_test, percentile)
        hybrid_time = time.time() - start_time
        
        # Evaluate hybrid model
        hybrid_metrics = evaluate_distributions(hybrid_totals, bootstrap_samples, test_claims, f"Hybrid {percentile}%")
        hybrid_metrics["time"] = hybrid_time
        hybrid_metrics["percentile"] = percentile
        hybrid_metrics["empirical_pct"] = percentile
        hybrid_metrics["theoretical_pct"] = 100 - percentile
        hybrid_metrics["totals"] = hybrid_totals  # Store totals for later visualization
        
        # Save detailed information
        hybrid_info[model_name] = hybrid_details
        hybrid_results[model_name] = hybrid_metrics
    
    # Create visualization comparing all hybrid models
    plt.figure(figsize=(14, 8))
    
    # Plot kernel density estimates
    sns.kdeplot(bootstrap_samples, label="Test Bootstrap", alpha=0.7, linestyle="--", color="black")
    sns.kdeplot(empirical_totals, label="Empirical", alpha=0.7, color="blue")
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    for i, percentile in enumerate(percentiles):
        model_name = f"hybrid_{percentile}"
        model_totals = hybrid_results[model_name].get("totals", [])
        if len(model_totals) > 0:
            sns.kdeplot(model_totals, label=f"Hybrid {percentile}%", alpha=0.7, color=colors[i])
    
    plt.axvline(test_total, color='r', linestyle='--', label=f"Actual Test Total: {test_total:.1f}M")
    plt.title("Distribution of Simulated Total Claims with Different Hybrid Percentages")
    plt.xlabel("Total Claim Amount (in millions)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("hybrid_percentile_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create a summary table
    print("\n=== HYBRID PERCENTILE COMPARISON ===")
    print("-" * 90)
    print(f"{'Model':<15} {'Mean Rel Error':<15} {'Std Rel Error':<15} {'Q99 Rel Error':<15} {'KL Div':<15} {'Wasserstein':<15}")
    print("-" * 90)
    
    # First print empirical baseline
    print(f"{'empirical':<15} {abs(results['empirical']['mean_rel_error'])*100:<15.2f} {abs(results['empirical']['std_rel_error'])*100:<15.2f} {abs(results['empirical']['q99_rel_error'])*100:<15.2f} {results['empirical']['kl_full']:<15.4f} {results['empirical']['wasserstein']:<15.2f}")
    
    # Then print all hybrid models
    for percentile in percentiles:
        model_name = f"hybrid_{percentile}"
        metrics = hybrid_results[model_name]
        print(f"{model_name:<15} {abs(metrics['mean_rel_error'])*100:<15.2f} {abs(metrics['std_rel_error'])*100:<15.2f} {abs(metrics['q99_rel_error'])*100:<15.2f} {metrics['kl_full']:<15.4f} {metrics['wasserstein']:<15.2f}")
    
    # Find the best hybrid model
    best_hybrid = None
    best_score = float('inf')
    for model_name, metrics in hybrid_results.items():
        # Calculate a balanced score (lower is better)
        score = (abs(metrics['mean_rel_error']) * 0.2 + 
                abs(metrics['std_rel_error']) * 0.2 + 
                abs(metrics['q99_rel_error']) * 0.2 + 
                metrics['kl_full'] * 0.2 + 
                metrics['wasserstein']/100 * 0.2)
        
        if score < best_score:
            best_score = score
            best_hybrid = model_name
    
    print("\nBest hybrid model:", best_hybrid.upper())
    
    # Save results to CSV
    result_df = pd.DataFrame(columns=["Model", "Mean Rel Error (%)", "Std Rel Error (%)", 
                                     "Q99 Rel Error (%)", "KL Divergence", "Wasserstein"])
    
    # Add empirical baseline
    result_df.loc[0] = ["empirical", 
                      abs(results['empirical']['mean_rel_error'])*100,
                      abs(results['empirical']['std_rel_error'])*100,
                      abs(results['empirical']['q99_rel_error'])*100,
                      results['empirical']['kl_full'],
                      results['empirical']['wasserstein']]
    
    # Add hybrid models
    for i, percentile in enumerate(percentiles):
        model_name = f"hybrid_{percentile}"
        metrics = hybrid_results[model_name]
        result_df.loc[i+1] = [model_name, 
                           abs(metrics['mean_rel_error'])*100,
                           abs(metrics['std_rel_error'])*100,
                           abs(metrics['q99_rel_error'])*100,
                           metrics['kl_full'],
                           metrics['wasserstein']]
    
    # Save to CSV
    result_df.to_csv("hybrid_percentile_comparison.csv", index=False)
    
    return hybrid_results, hybrid_info

def plot_model_comparison(test_claims, empirical_results, hybrid_results, theoretical_results, save_path="model_comparison_cdf.png"):
    """
    Create a plot comparing the CDF of the different modeling approaches against actual test data.
    
    Args:
        test_claims: The actual test claims (ground truth)
        empirical_results: Results from the empirical model
        hybrid_results: Results from the best hybrid model
        theoretical_results: Results from the theoretical model
        save_path: Path to save the resulting plot
    """
    # First, generate bootstrap samples from the test claims to create a proper distribution
    test_size = len(test_claims)
    bootstrap_size = 10000
    bootstrap_samples = []
    for _ in range(bootstrap_size):
        sampled_claims = np.random.choice(test_claims, size=test_size, replace=True)
        bootstrap_samples.append(np.sum(sampled_claims))
    bootstrap_samples = np.array(bootstrap_samples)
    
    # Get the actual total for the test set (for reference)
    actual_total = np.sum(test_claims)
    print(f"Test set actual total: {actual_total:.2f}M")
    print(f"Bootstrap mean: {np.mean(bootstrap_samples):.2f}M, std: {np.std(bootstrap_samples):.2f}M")
    
    # Get the model results
    empirical_totals = empirical_results["totals"]
    hybrid_totals = hybrid_results["totals"]
    theoretical_totals = theoretical_results["totals"]
    
    # Create density comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot histograms with density=True and alpha for transparency
    plt.hist(bootstrap_samples, bins=30, density=True, alpha=0.3, label='Test Bootstrap')
    plt.hist(empirical_totals, bins=30, density=True, alpha=0.3, label='Empirical Model')
    plt.hist(hybrid_totals, bins=30, density=True, alpha=0.3, label='Hybrid Model (97/3)')
    plt.hist(theoretical_totals, bins=30, density=True, alpha=0.3, label='Theoretical Model')
    
    # Add a vertical line for the actual test total
    plt.axvline(x=actual_total, color='k', linestyle='--', linewidth=2, label=f'Test Actual Total: {actual_total:.1f}M')
    
    # Add labels and title
    plt.title("Density Comparison of Different Modeling Approaches")
    plt.xlabel("Total Claim Amount (in millions)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the density plot
    plt.savefig("model_comparison_density.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create CDF comparison plot
    plt.figure(figsize=(12, 8))
    
    # Calculate CDFs
    def calc_cdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y
    
    # Get CDF values
    x_test, y_test = calc_cdf(bootstrap_samples)
    x_emp, y_emp = calc_cdf(empirical_totals)
    x_hyb, y_hyb = calc_cdf(hybrid_totals)
    x_theo, y_theo = calc_cdf(theoretical_totals)
    
    # Plot CDFs
    plt.plot(x_test, y_test, 'k-', linewidth=2, label='Test Bootstrap')
    plt.plot(x_emp, y_emp, 'b-', linewidth=1.5, label='Empirical Model')
    plt.plot(x_hyb, y_hyb, 'r-', linewidth=1.5, label='Hybrid Model (97/3)')
    plt.plot(x_theo, y_theo, 'g-', linewidth=1.5, label='Theoretical Model')
    
    # Add a vertical line for the actual test total
    plt.axvline(x=actual_total, color='k', linestyle='--', linewidth=1, 
              label=f'Test Actual Total: {actual_total:.1f}M')
    
    # Add labels and title
    plt.title("CDF Comparison of Different Modeling Approaches")
    plt.xlabel("Total Claim Amount (in millions)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the CDF plot
    plt.savefig("model_comparison_cdf.png", dpi=300, bbox_inches="tight")
    
    # Create quantile plot
    plt.figure(figsize=(12, 8))
    
    quantiles = np.arange(0.01, 1.0, 0.01)
    
    test_quantiles = np.quantile(bootstrap_samples, quantiles)
    empirical_quantiles = np.quantile(empirical_totals, quantiles)
    hybrid_quantiles = np.quantile(hybrid_totals, quantiles)
    theoretical_quantiles = np.quantile(theoretical_totals, quantiles)
    
    plt.plot(quantiles, test_quantiles, 'k-', linewidth=2, label='Test Bootstrap')
    plt.plot(quantiles, empirical_quantiles, 'b-', linewidth=1.5, label='Empirical Model')
    plt.plot(quantiles, hybrid_quantiles, 'r-', linewidth=1.5, label='Hybrid Model (97/3)')
    plt.plot(quantiles, theoretical_quantiles, 'g-', linewidth=1.5, label='Theoretical Model')
    
    # Add horizontal line for actual test total
    plt.axhline(y=actual_total, color='k', linestyle='--', linewidth=1, 
             label=f'Test Actual Total: {actual_total:.1f}M')
    
    plt.title("Quantile Comparison of Different Modeling Approaches")
    plt.xlabel("Quantile")
    plt.ylabel("Total Claim Amount (in millions)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the quantile plot
    plt.savefig("model_comparison_quantile.png", dpi=300, bbox_inches="tight")
    
    print("Model comparison plots created and saved.")

def main():
    """Main function to run the analysis."""
    # Run model comparison with both split types
    # Comment out one type if you only want to run that particular experiment
    
    # Compare hybrid percentages
    percentiles_to_test = [94.0, 95.0, 96.0, 97.0, 98.0, 99.0]
    compare_hybrid_percentages(percentiles_to_test)
    
    # Add code to perform modeling comparison and generate the plot
    print("\nGenerating model comparison plots...")
    
    # Set a seed for consistency
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    claims = load_data()
    
    # Create a simple random split manually instead of using the function
    print("Creating train-test split...")
    claims_array = claims["Loss"].values
    n = len(claims_array)
    train_size = int(0.5 * n)
    
    # Shuffle the data
    indices = np.random.permutation(n)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_claims = claims_array[train_indices]
    test_claims = claims_array[test_indices]
    
    print(f"Train set: {len(train_claims)} claims")
    print(f"Test set: {len(test_claims)} claims")
    
    # Get the actual total for the test set
    actual_total = np.sum(test_claims)
    print(f"Actual test set total: {actual_total:.2f}M")
    
    # Number of simulations
    num_simulations = 10000
    test_size = len(test_claims)
    
    # Perform simulations with different models
    print("Simulating with empirical model...")
    empirical_totals = []
    for _ in range(num_simulations):
        sample = np.random.choice(train_claims, size=test_size, replace=True)
        empirical_totals.append(np.sum(sample))
    empirical_results = {"totals": np.array(empirical_totals)}
    
    print("Simulating with hybrid model (97/3)...")
    threshold_percentile = 97.0
    threshold = np.percentile(train_claims, threshold_percentile)
    non_extreme = train_claims[train_claims <= threshold]
    extreme = train_claims[train_claims > threshold]
    
    print(f"  Hybrid {threshold_percentile}%: Split at {threshold:.2f}M, {len(extreme)} extreme claims ({len(extreme)/len(train_claims):.1%})")
    
    # Find best distribution for extreme claims
    from scipy import stats
    
    distributions = [stats.lognorm, stats.pareto, stats.genpareto]
    best_dist = None
    best_params = None
    best_sse = np.inf
    
    for dist in distributions:
        try:
            params = dist.fit(extreme)
            sse = np.sum((dist.cdf(sorted(extreme), *params) - np.arange(1, len(extreme) + 1) / len(extreme)) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_dist = dist
                best_params = params
        except Exception as e:
            print(f"    Error fitting {dist.name}: {e}")
    
    print(f"    Best distribution for extreme claims: {best_dist.name}")
    
    # Generate hybrid samples
    hybrid_totals = []
    p_extreme = 1 - threshold_percentile/100
    
    for _ in range(num_simulations):
        claim_types = np.random.choice(['non-extreme', 'extreme'], size=test_size, p=[1-p_extreme, p_extreme])
        claims = []
        
        for claim_type in claim_types:
            if claim_type == 'non-extreme':
                claims.append(np.random.choice(non_extreme))
            else:
                claim = best_dist.rvs(*best_params, size=1)[0]
                claims.append(max(claim, threshold))  # Ensure it's above threshold
        
        hybrid_totals.append(np.sum(claims))
    
    hybrid_results = {"totals": np.array(hybrid_totals)}
    
    print("Simulating with theoretical model...")
    # Find best overall distribution
    best_overall_dist = None
    best_overall_params = None
    best_overall_sse = np.inf
    
    for dist in [stats.lognorm, stats.genpareto, stats.gamma, stats.weibull_min]:
        try:
            params = dist.fit(train_claims)
            sse = np.sum((dist.cdf(sorted(train_claims), *params) - np.arange(1, len(train_claims) + 1) / len(train_claims)) ** 2)
            if sse < best_overall_sse:
                best_overall_sse = sse
                best_overall_dist = dist
                best_overall_params = params
        except Exception as e:
            print(f"    Error fitting {dist.name}: {e}")
    
    print(f"    Best overall distribution: {best_overall_dist.name}")
    
    # Generate theoretical samples
    theoretical_totals = []
    
    for _ in range(num_simulations):
        claims = best_overall_dist.rvs(*best_overall_params, size=test_size)
        claims = np.maximum(claims, 0)  # Ensure non-negative
        theoretical_totals.append(np.sum(claims))
    
    theoretical_results = {"totals": np.array(theoretical_totals)}
    
    # Create the comparison plots
    plot_model_comparison(test_claims, empirical_results, hybrid_results, theoretical_results)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main() 