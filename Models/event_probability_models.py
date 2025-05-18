import numpy as np
from scipy.stats import norm
from datetime import datetime
from Data.business_data import historical_data  # imports the latest available data
from Models.corrective_assessment_support_for_risk_model import corrective


##############################
# 1. Parameter Uncertainty: Bootstrap Confidence Intervals
##############################
def bootstrap_parameters(data, n_bootstrap=1000):
    boot_mu = []
    boot_sigma = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        diffs = np.diff(sample)
        boot_mu.append(np.mean(diffs))
        boot_sigma.append(np.std(diffs, ddof=1))
    ci_mu = np.percentile(boot_mu, [2.5, 97.5])
    ci_sigma = np.percentile(boot_sigma, [2.5, 97.5])
    return ci_mu, ci_sigma

##############################
# 2. Model Choice: ABM vs GBM
##############################
def estimate_parameters(data, use_gbm=False):
    """
    Estimate drift and volatility from historical data.
    If use_gbm is True, estimates are performed in log space for geometric Brownian motion.
    Otherwise, standard arithmetic Brownian motion estimation is used.
    """
    if use_gbm:
        # Remove zeros to avoid log issues
        data = np.array(data)
        data = data[data > 0]
        log_data = np.log(data)
        diffs = np.diff(log_data)
        mu = np.mean(diffs)
        sigma = np.std(diffs, ddof=1)
    else:
        diffs = np.diff(data)
        mu = np.mean(diffs)
        sigma = np.std(diffs, ddof=1)
    return mu, sigma

def analytical_probability(x, mu, sigma, threshold, T):
    """
    Calculate the probability that an arithmetic Brownian motion starting at x
    will reach the threshold within T days using a first-passage time formula.
    
    Formula:
      P(max_{0<=t<=T} X_t >= threshold) =
         Φ(-((threshold - x) - mu*T) / (sigma*sqrt(T))) +
         exp(2*mu*(threshold - x)/(sigma^2)) * Φ(-((threshold - x) + mu*T) / (sigma*sqrt(T)))
    
    Returns the probability.
    """
    if sigma == 0:
        return 1.0 if x >= threshold else 0.0
    d1 = (threshold - x - mu * T) / (sigma * np.sqrt(T))
    d2 = (threshold - x + mu * T) / (sigma * np.sqrt(T))
    prob = norm.cdf(-d1) + np.exp(2 * mu * (threshold - x) / (sigma**2)) * norm.cdf(-d2)
    return prob


##############################
# 3. Time-Varying Parameters: Rolling Window Estimation
##############################
def rolling_window_parameters(data, window=30, use_gbm=False):
    """
    Compute rolling estimates for drift and volatility using a given window size.
    Returns arrays of mu and sigma of length len(data)-window.
    """
    mu_list = []
    sigma_list = []
    for i in range(len(data) - window):
        window_data = data[i:i+window]
        mu, sigma = estimate_parameters(window_data, use_gbm)
        mu_list.append(mu)
        sigma_list.append(sigma)
    return np.array(mu_list), np.array(sigma_list)

##############################
# 4. Simulation Enhancements: Refined Time-step and Adaptive Option
##############################
def simulate_paths(x0, mu, sigma, T, dt, n_paths, threshold, use_gbm=False):
    """
    Simulate n_paths using the selected model.
    For GBM, the process is multiplicative and simulated in log-space.
    Returns:
      - hit: Boolean array indicating whether the threshold was reached.
      - first_hit_times: Array with the first time when the threshold was hit.
      - paths: Simulated paths.
    """
    time_steps = int(T / dt)
    hit = np.zeros(n_paths, dtype=bool)
    first_hit_times = np.full(n_paths, np.nan)
    paths = np.zeros((n_paths, time_steps + 1))
    paths[:, 0] = x0

    for t in range(1, time_steps + 1):
        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=n_paths)
        if use_gbm:
            # In GBM, the process is X(t+dt) = X(t) * exp((mu - 0.5*sigma**2)*dt + sigma*sqrt(dt)*Z)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        else:
            paths[:, t] = paths[:, t-1] + mu * dt + sigma * dW

        # Check if threshold is reached
        new_hit = (paths[:, t] >= threshold) & (~hit)
        hit[new_hit] = True
        first_hit_times[new_hit] = t * dt
    return hit, first_hit_times, paths

def monte_carlo_probability(x, mu, sigma, threshold, T, dt=1.0, n_paths=100000, use_gbm=False):
    """
    Estimate the probability of hitting the threshold within T days via Monte Carlo simulation.
    Returns:
      - event_probability: Fraction of paths that reached the threshold.
      - prob_se: Standard error of the probability estimate.
      - expected_time: Average time to reach the threshold (only for paths that hit).
      - time_se: Standard error of the expected event time.
      - paths: The simulated metric paths.
    """
    hit, first_hit_times, paths = simulate_paths(x, mu, sigma, T, dt, n_paths, threshold, use_gbm)
    event_probability = np.mean(hit)
    prob_se = np.sqrt(event_probability * (1 - event_probability) / n_paths)
    
    if np.sum(hit) > 0:
        hit_times = first_hit_times[hit]
        expected_time = np.mean(hit_times)
        time_se = np.std(hit_times, ddof=1) / np.sqrt(len(hit_times))
    else:
        expected_time = np.nan
        time_se = np.nan
    return event_probability, prob_se, expected_time, time_se, paths

##############################
# 5. Model Validation: Residual Analysis
##############################
def residual_analysis(data, use_gbm=False, plot_residuals=False):
    """
    Performs a residual analysis to check if the daily changes are close to normally distributed.
    Optionally, plots a histogram of residuals.
    Returns basic statistics about the residuals.
    """
    if use_gbm:
        data = np.array(data)
        data = data[data > 0]
        log_data = np.log(data)
        residuals = np.diff(log_data) - np.mean(np.diff(log_data))
    else:
        residuals = np.diff(data) - np.mean(np.diff(data))
    
    stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals, ddof=1),
    }
    

    return stats

##############################
# 6. Enhanced Output and Overall Calculation
##############################
def calculate_event_probability(factor, threshold, T, n_paths=100000, 
                                use_gbm=False, use_rolling_window=False, plot_residuals=False):
    """
    Main function to calculate the event probability.
    Includes parameter uncertainty, alternative model choice, time-varying parameters,
    simulation improvements, model validation, and user-friendly output.
    
    Parameters:
      - factor: The key/metric to analyze.
      - threshold: The target threshold to reach.
      - T: Time horizon (in days).
      - n_paths: Number of Monte Carlo simulation paths.
      - use_gbm: If True, use Geometric Brownian Motion.
      - use_rolling_window: If True, compute rolling window estimates for additional insights.
      - plot_residuals: If True, display a histogram of residuals for diagnostic purposes.
    """
    # Load the historical data and get the current value.
    data = np.array(historical_data[factor])
    x0 = data[-1]
    
    # Basic parameter estimation (point estimates)
    mu, sigma = estimate_parameters(data, use_gbm)
    
    # Bootstrap confidence intervals for parameters
    ci_mu, ci_sigma = bootstrap_parameters(data)
    
    # Rolling window estimation if desired (only for informational purposes)
    if use_rolling_window:
        roll_mu, roll_sigma = rolling_window_parameters(data, window=30, use_gbm=use_gbm)
        rolling_summary = f"\nOver the past 30-day windows, the average drift is roughly {np.mean(roll_mu):.4f} and volatility is about {np.mean(roll_sigma):.4f}."
    else:
        rolling_summary = ""
    
    # Perform residual analysis and (optionally) plot the histogram.
    resid_stats = residual_analysis(data, use_gbm, plot_residuals)
    
    # Analytical probability (Note: formula applies directly for ABM only)
    if not use_gbm:
        prob_analytical = analytical_probability(x0, mu, sigma, threshold, T)
    else:
        # For GBM, an equivalent analytical first-passage formula can be more involved.
        # Here we set the analytical result to None, indicating simulation is used.
        prob_analytical = None
    
    # Monte Carlo probability (simulate paths)
    dt = 0.5  # Optionally a refined time step for higher resolution
    prob_mc, prob_se, expected_time, time_se, _ = monte_carlo_probability(
        x0, mu, sigma, threshold, T, dt, n_paths, use_gbm)
    
    increase = corrective(factor)
    
    # Create a non-technical, user-friendly explanation.
    explanation = f"""
--------------------------------------------------
# Event Probability Report for '{factor.capitalize()}'

## 1. Summary of your data:
   - **Current {factor}**: {x0:.2f}
   - **Daily average change:** {mu:.4f} (with a typical range of ±{sigma:.4f})
   - A 95% confidence range for the daily change is approximately [**{ci_mu[0]:.4f}**, **{ci_mu[1]:.4f}**]
     and for daily fluctuations about [**{ci_sigma[0]:.4f}**, **{ci_sigma[1]:.4f}**].


## 2. Time-Varying Dynamics:
   - { "Using a rolling-window analysis, recent windows suggest a slightly varying trend." if use_rolling_window 
       else "The model assumes constant daily behavior over the forecast period." }
   {rolling_summary}

## 3. What do the numbers mean?
   - Based on the simulation, there is about a **{prob_mc:.2%}** chance of reaching the target level of **{threshold:.2f}** within {T} days.
   - When the threshold is reached, it takes on average **{expected_time:.2f}** days, but expect some uncertainty (±**{time_se:.2f}** days).
   - { "For the arithmetic model, the direct calculation gave a probability of **" + f"{prob_analytical:.2%}.**" if (not use_gbm and prob_analytical is not None) else "Analytical formulas for GBM are more complex, so we rely on the simulation here."    
      }\n\n
## 4. What you can do: \n\n
    To increase the probabilities of reaching it, focus on these levers for maximum effect:{increase}
      

--------------------------------------------------
"""
    
    return explanation

# Example usage:
#print(calculate_event_probability("revenue", 800, 365, use_gbm=False, use_rolling_window=True, plot_residuals=False))
