import numpy as np
from Data.business_data import historical_data
from scipy import stats

def business_twin(factor, rate):
    """
    Generate future business trends with a forced increase in the driver factor.
    Computes percentage changes per factor and per cluster, builds a unified business profile,
    and outputs a detailed narrative summary.
    """
    main = factor  # chosen driver

    # -------------------------------------------------------------------------
    # 1. Enhanced Fourier-based Filtering for Trend Extraction
    #    (Adaptive frequency filtering based on cumulative energy)
    # -------------------------------------------------------------------------
    def filter_signal(signal, energy_threshold=0.99):
        fft_signal = np.fft.fft(signal)
        N = len(fft_signal)
        energy = np.abs(fft_signal) ** 2
        total_energy = np.sum(energy)
        # Sort frequencies by energy
        sorted_indices = np.argsort(energy)[::-1]  
        cumulative_energy = np.cumsum(energy[sorted_indices]) / total_energy
        # Find how many frequencies capture the desired energy_threshold
        num_keep = np.searchsorted(cumulative_energy, energy_threshold) + 1
        # Create a mask that preserves these highest-energy coefficients
        mask = np.zeros(N, dtype=bool)
        indices_to_keep = sorted_indices[:num_keep]
        mask[indices_to_keep] = True
        filtered_fft = np.where(mask, fft_signal, 0)
        return np.real(np.fft.ifft(filtered_fft))

    # Compute filtered historical signals.
    filtered_hist = {fctr: filter_signal(data) for fctr, data in historical_data.items()}

    # -------------------------------------------------------------------------
    # 2. Define Clusters for Multidimensional Sensitivity (Manual + Weighted)
    # -------------------------------------------------------------------------
    clusters = {
        "market_performance": ["sales", "revenue", "market_share", "brand_awareness"],
        "customer_engagement": ["customer_sat", "website_traffic", "digital_engagement", "social_media_presence"],
        "operational": ["employee_productivity", "operational_efficiency", "inventory", "operational_costs", "financial_health"],
    }
    # Compute weights for each factor based on inverse variance (more stable = higher weight).
    factor_weights = {}
    for fctr, data in historical_data.items():
        var = np.var(data, ddof=1)
        factor_weights[fctr] = 1.0 / (var + 1e-6)

    # -------------------------------------------------------------------------
    # 3. Robust Nonlinear Sensitivity Estimation Using Theil–Sen Regression
    # -------------------------------------------------------------------------
    def compute_robust_sensitivity(y, x):
        # The Theil–Sen estimator is robust to outliers.
        slope, intercept, lower, upper = stats.theilslopes(y, x, 0.95)
        # Use the slope as the sensitivity; here using the midpoint of the confidence interval is optional.
        sensitivity = slope  
        # Also store the robust coefficients (slope and intercept) as a fallback.
        return sensitivity, (slope, intercept)
    
    learned_sensitivities = {}
    robust_coefs = {}
    driver_data = filtered_hist[main]
    for fctr, data in filtered_hist.items():
        if fctr == main:
            learned_sensitivities[fctr] = 1.0  # baseline for driver
            robust_coefs[fctr] = (1.0, 0.0)      # dummy coefficients for driver
        else:
            sens, coefs = compute_robust_sensitivity(data, driver_data)
            learned_sensitivities[fctr] = sens
            robust_coefs[fctr] = coefs

    # Also build a composite dependency for future adjustments.
    # Here, we combine the estimated sensitivity (absolute value) and the composite correlation.
    def composite_dependency(fctr):
        # Compute Pearson and Spearman correlations between fctr and driver.
        driver = filtered_hist[main]
        series = filtered_hist[fctr]
        if np.std(driver) < 1e-6 or np.std(series) < 1e-6:
            comp_corr = 0.0
        else:
            pearson = np.corrcoef(driver, series)[0, 1]
            spearman = stats.spearmanr(driver, series)[0]
            comp_corr = (pearson + spearman) / 2.0
        # Combine with sensitivity: here we take the average of the absolute sensitivity and comp_corr
        return (abs(learned_sensitivities[fctr]) + comp_corr) / 2.0

    composite_dependencies = {fctr: composite_dependency(fctr)
                              for fctr in learned_sensitivities if fctr != main}

    # Aggregate and print cluster sensitivities using weighted averages
    cluster_sensitivities = {}
    for cluster, factors_list in clusters.items():
        sens_vals = []
        weights = []
        for f in factors_list:
            if f in learned_sensitivities:
                sens_vals.append(learned_sensitivities[f])
                weights.append(factor_weights.get(f, 1.0))
        if sens_vals:
            cluster_sensitivities[cluster] = np.average(sens_vals, weights=weights)
        else:
            cluster_sensitivities[cluster] = 0

    # -------------------------------------------------------------------------
    # 4. Generate Future Data Using Quadratic Regression and Multiplicative Adjustments
    #    (Adjusted with composite dependencies)
    # -------------------------------------------------------------------------
    t_future = np.arange(730, 800)
    def generate_future_data(t_future, filtered_hist, sensitivities, main_factor=main, rate=0.0):
        future_data = {}
        poly_coefs = {}
        for fctr, trend in filtered_hist.items():
            x_hist = np.arange(len(trend))
            coefs = np.polyfit(x_hist, trend, 2)
            poly_coefs[fctr] = coefs

        driver_coefs = poly_coefs[main_factor]
        driver_baseline_forecast = np.polyval(driver_coefs, t_future)
        forced_driver_forecast = driver_baseline_forecast * (1.0 + rate)

        for fctr, coefs in poly_coefs.items():
            baseline_forecast = np.polyval(coefs, t_future)
            if fctr == main_factor:
                future_data[fctr] = baseline_forecast * (1.0 + rate)
            else:
                # Use the composite dependency for adjustment
                comp_dep = composite_dependencies.get(fctr, 1.0)
                adjustment_factor = (forced_driver_forecast / driver_baseline_forecast) ** comp_dep
                future_data[fctr] = baseline_forecast * adjustment_factor
        return future_data

    # Compute future forecasts for the baseline scenario and the increased driver scenario.
    baseline_future = generate_future_data(t_future, filtered_hist, learned_sensitivities, main_factor=main, rate=0.0)
    increased_future = generate_future_data(t_future, filtered_hist, learned_sensitivities, main_factor=main, rate=rate)

    # -------------------------------------------------------------------------
    # 5. Filter Future Signals to Extract Underlying Trends
    # -------------------------------------------------------------------------
    def filter_future_signals(future_dict):
        return {fctr: filter_signal(signal) for fctr, signal in future_dict.items()}

    baseline_filtered = filter_future_signals(baseline_future)
    increased_filtered = filter_future_signals(increased_future)

    # -------------------------------------------------------------------------
    # 6. Build Unified Profiles via Averaging Across Factors
    # -------------------------------------------------------------------------
    def build_unified_profile(filtered_dict):
        signals = np.stack(list(filtered_dict.values()), axis=0)
        return np.mean(signals, axis=0)

    unified_baseline = build_unified_profile(baseline_filtered)
    unified_increased = build_unified_profile(increased_filtered)

    baseline_profile_avg = np.mean(unified_baseline)
    increased_profile_avg = np.mean(unified_increased)
    business_profile_change = (increased_profile_avg - baseline_profile_avg) / baseline_profile_avg * 100

    if business_profile_change >= 0:
        profile_statement = (
            f"Overall, increasing {main} by {rate*100:.0f}% leads to an average "
            f"increase of {business_profile_change:.2f}% in the unified business profile. "
            "This suggests a broadly positive impact across key performance areas."
        )
    else:
        profile_statement = (
            f"Overall, increasing {main} by {rate*100:.0f}% leads to an average "
            f"decrease of {business_profile_change:.2f}% in the unified business profile. "
            "This may signal potential challenges that need to be addressed."
        )

    # -------------------------------------------------------------------------
    # 7. Compare Each Factor’s Future Trends: Baseline vs. Increased Driver
    # -------------------------------------------------------------------------
    factors = list(historical_data.keys())
    baseline_avgs = [np.mean(baseline_filtered[fctr]) for fctr in factors]
    increased_avgs = [np.mean(increased_filtered[fctr]) for fctr in factors]
    percentage_changes = [((inc - base) / base * 100) for inc, base in zip(increased_avgs, baseline_avgs)]

    comparison_lines = []
    comparison_lines.append(f"\n=== FUTURE SCENARIO COMPARISON: Effects of Increased {main.capitalize()} Spend ===")
    for i, fctr in enumerate(factors):
        comparison_lines.append(f"{fctr:20s} | Change: {percentage_changes[i]:6.2f}%")
    comparison_lines.append("-------------------------------------------------------------------------------\n")
    comparison_summary = "\n".join(comparison_lines)

    # -------------------------------------------------------------------------
    # 8. Compute Cluster Performance Percentage Changes
    # -------------------------------------------------------------------------
    cluster_baseline = {}
    cluster_increased = {}
    cluster_percentage_changes = {}
    for cluster, factors_list in clusters.items():
        baseline_vals = []
        increased_vals = []
        for f in factors_list:
            if f in baseline_filtered:
                baseline_vals.append(np.mean(baseline_filtered[f]))
                increased_vals.append(np.mean(increased_filtered[f]))
        if baseline_vals:
            base_avg = np.mean(baseline_vals)
            inc_avg = np.mean(increased_vals)
            cluster_baseline[cluster] = base_avg
            cluster_increased[cluster] = inc_avg
            cluster_percentage_changes[cluster] = (inc_avg - base_avg) / base_avg * 100

    cluster_lines = []
    for cluster in clusters:
        change = cluster_percentage_changes.get(cluster, 0.0)
        cluster_lines.append(f"{cluster:20s} | Change: {change:6.2f}%\n\n")
    cluster_lines.append("-------------------------------------------------------------------------------\n")
    cluster_summary = "\n".join(cluster_lines)
    
    # -------------------------------------------------------------------------
    # 9. Build Enhanced Narrative Summary Output
    # -------------------------------------------------------------------------
    narrative_summary = (
        "-----------------------------\n\n"
        "# Business Twin Factor Simulation\n\n"
        f"Driver Factor: **{main.capitalize()}**\n\n"
        f"Rate: {rate*100:.0f}%\n\n"
        f"Average Business Profile Change: {business_profile_change:.2f}%\n\n\n "

        "## Future Scenario Comparison (Individual Factors):\n"
        "------------------------------\n"
        "Below is a detailed comparison of the baseline forecast versus the forecast when the driver factor is increased:\n\n"
        "#### Factor                 | Change (%)\n"
    )
    for i, fctr in enumerate(factors):
        narrative_summary += f"{fctr.replace('_', ' '):20s} | {percentage_changes[i]:6.2f}%\n\n"
    narrative_summary += (
        "--------------------------------------\n\n"
        "### 3. Cluster Performance Changes:\n"
        "------------------------------\n"
        f"{cluster_summary}\n"
        "### 4. Strategic Implications:\n"
        "--------------------------\n"
        "• The driver factor substantially influences overall business performance.\n\n"
        "• Combined sensitivity measures suggest that other factors respond not only in magnitude but also in pace to changes in the driver.\n\n"
        "• Unified and cluster-based profiles reveal where the strongest impacts occur.\n\n"
        "• Positive changes indicate promising strategic opportunities, while negative changes highlight areas that may need targeted intervention.\n\n"
        "Overall, these insights offer a data-driven foundation for resource allocation and long-term strategic planning.\n\n"
    )
    narrative_summary += profile_statement



    return narrative_summary

