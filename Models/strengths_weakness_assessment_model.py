import numpy as np

from Data.business_data import historical_data

# ---------------------------
# Define the full set of factors.
# ---------------------------
impact = ["sales", "revenue", "marketing", "customer_sat", "website_traffic", "employee_productivity",
          "operational_efficiency", "competitive_advantage", "inventory", "customer_loyalty", "brand_awareness",
          "cost_efficiency", "supply_chain_reliability", "innovation", "employee_satisfaction", "market_share",
          "digital_engagement", "social_media_presence", "product_quality", "operational_costs", "financial_health"]

base_value = 10000
rate = 0.10  # Forced increase rate for the driver candidate

# ---------------------------
# Simulation settings: outcomes we want to drive.
# ---------------------------

# =============================================================================

# =============================================================================
# 2. Define a Fourier-based Filtering Function for Trend Extraction
# =============================================================================
def filter_signal(signal, keep_ratio=0.01):
    fft_signal = np.fft.fft(signal)
    N = len(fft_signal)
    filtered_fft = np.zeros_like(fft_signal)
    filtered_fft[0] = fft_signal[0]  # Preserve DC component
    center = N // 2
    half_keep = int((keep_ratio * N) / 2)
    filtered_fft[center - half_keep:center + half_keep] = fft_signal[center - half_keep:center + half_keep]
    return np.real(np.fft.ifft(filtered_fft))

# Pre-compute the filtered historical signals.
filtered_hist = {factor: filter_signal(data) for factor, data in historical_data.items()}

# =============================================================================
# 3. Learn Sensitivities from Historical Data Using Linear Regression
# =============================================================================
def compute_sensitivity(y, x):
    beta, intercept = np.polyfit(x, y, 1)
    return beta

# =============================================================================
# 4. Generate Future Data Using Quadratic Regression and Multiplicative Adjustments
# =============================================================================
t_future = np.arange(730, 800)
def generate_future_data(t_future, filtered_hist, sensitivities, main_factor, rate=0.0):
    """
    For each factor, fit a quadratic model on the historical (filtered) trend.
    For the driver candidate (main_factor), force an increase of (1 + rate).
    For other factors, adjust the baseline forecast multiplicatively:
         future = baseline * (forced_driver / baseline_driver)^β
    where β is the learned sensitivity.
    """
    future_data = {}
    poly_coefs = {}
    # Fit quadratic models for each factor.
    for factor, trend in filtered_hist.items():
        x_hist = np.arange(len(trend))
        coefs = np.polyfit(x_hist, trend, 2)
        poly_coefs[factor] = coefs

    # Compute the candidate driver's baseline forecast.
    driver_coefs = poly_coefs[main_factor]
    driver_baseline_forecast = np.polyval(driver_coefs, t_future)
    forced_driver_forecast = driver_baseline_forecast * (1.0 + rate)

    for factor, coefs in poly_coefs.items():
        baseline_forecast = np.polyval(coefs, t_future)
        if factor == main_factor:
            future_data[factor] = baseline_forecast * (1.0 + rate)
        else:
            beta = sensitivities[factor]
            adjustment_factor = (forced_driver_forecast / driver_baseline_forecast) ** beta
            future_data[factor] = baseline_forecast * adjustment_factor
    return future_data

# =============================================================================
# 5. (Optional) Filter Future Signals to Extract Underlying Trends
# =============================================================================
def filter_future_signals(future_dict):
    return {factor: filter_signal(signal) for factor, signal in future_dict.items()}

# =============================================================================
# 6. Build Unified Profiles via Averaging
# =============================================================================
def build_unified_profile(filtered_dict):
    signals = np.stack(list(filtered_dict.values()), axis=0)
    return np.mean(signals, axis=0)

# =============================================================================
# 7. Define find_weaknesses Function
# =============================================================================
def find_strengths(outcome):
    """
    For a given outcome factor (e.g., "sales" or "revenue"), this function
    computes the effect of forcing each candidate driver (all factors except the outcome itself).
    It then extracts the top 3 strengths and the bottom 3 weaknesses, and returns
    a summary paragraph describing the potential losses and strengths.
    """
    results_table = []
    # For each candidate driver (all factors except the outcome itself)
    for candidate in impact:
        if candidate.lower() == outcome.lower():
            continue

        # Compute learned sensitivities using candidate as the forced driver.
        candidate_sensitivities = {}
        driver_data = filtered_hist[candidate]
        for factor, data in filtered_hist.items():
            if factor == candidate:
                candidate_sensitivities[factor] = 1.0
            else:
                candidate_sensitivities[factor] = compute_sensitivity(data, driver_data)

        # Generate future data with candidate forced to increase.
        baseline_future = generate_future_data(t_future, filtered_hist, candidate_sensitivities, main_factor=candidate, rate=0.0)
        increased_future = generate_future_data(t_future, filtered_hist, candidate_sensitivities, main_factor=candidate, rate=rate)

        baseline_filtered = filter_future_signals(baseline_future)
        increased_filtered = filter_future_signals(increased_future)

        unified_baseline = build_unified_profile(baseline_filtered)
        unified_increased = build_unified_profile(increased_filtered)
        business_profile_change = (np.mean(unified_increased) - np.mean(unified_baseline)) / np.mean(unified_baseline) * 100

        # Instead of candidate's own change, measure the change in the outcome variable.
        baseline_outcome_avg = np.mean(baseline_filtered[outcome])
        increased_outcome_avg = np.mean(increased_filtered[outcome])
        outcome_change = (increased_outcome_avg - baseline_outcome_avg) / baseline_outcome_avg * 100

        financial_impact = base_value * (outcome_change / 100)

        if financial_impact > 0:
            financial_impact = financial_impact * 1
            if financial_impact > 1000:

                result = {
                    "name": candidate,
                    outcome: outcome_change,  # This is the driver effect on the outcome.
                    "unified_business_profile": business_profile_change,
                    "financial_impact": financial_impact 
                }
                #print(result["name"], result[outcome])
                results_table.append(result)

    # Sort candidates by the outcome change (driver effect) in descending order.
    sorted_by_effect = sorted(results_table, key=lambda x: x["financial_impact"] if x["financial_impact"] < 1000 else float('inf'), reverse=True)
    

    # Extract top 3 strengths and bottom 3 weaknesses.
    strengths = sorted_by_effect[:8]
    

    strengths_str = "\n".join([
        f"• **{entry['name']}** : {outcome.capitalize()} impact increased by {abs(entry[outcome]):.2f}%, with a increase of {abs(entry['unified_business_profile']):.2f}% in the business profile and a potential ROI gain of ${abs(entry['financial_impact']):.2f}.\n An ROI of above $1000 signifies a high probability gain. \n"
        for entry in strengths
    ])

    summary = (
        "-----------------------------\n\n"
        
        f"The AI Copilot continuously analyzes your key business drivers to provide real-time insights into your {outcome} performance. "
        f"By evaluating current data against a baseline investment of $1000, it dynamically identifies how each driver influences your overall business profile and revenue outcomes.\n\n"
        
        f"**Key Strengths:**\n\n"
        f"{strengths_str}\n\n"
        
        f"**Detailed Analysis:**\n"
        f"It continuously monitors these metrics, showing that their robust performance translates into significant improvements in both {outcome} and the integrated business profile. "
        f"This real-time analysis confirms strong market performance and highlights operational strategies that are effectively driving growth.\n"
        f"- **Capitalize on Strengths:** The Copilot advises leveraging and further investing in these top drivers to amplify their positive impact on {outcome}. "
        f"By allocating resources intelligently based on these insights, you can maximize financial returns and boost overall business health.\n\n"
        
        f"**Conclusion:**\n"
        f"Through strategic enhancements and by addressing identified vulnerabilities, the AI Copilot enables your business to achieve notable improvements in {outcome} performance and overall operational efficiency. "
        f"This balanced, data-driven approach not only promotes sustainable growth but also ensures that every strategic adjustment is aligned with your financial goals, driving higher returns in real time."
    )

    return summary


def find_weaknesses(outcome):
    
    """
    For a given outcome factor (e.g., "sales" or "revenue"), this function
    computes the effect of forcing each candidate driver (all factors except the outcome itself).
    It then extracts the top 3 strengths and the bottom 3 weaknesses, and returns
    a summary paragraph describing the potential losses and strengths.
    """
    results_table = []
    # For each candidate driver (all factors except the outcome itself)
    for candidate in impact:
        
        if candidate.lower() == outcome.lower():
            continue

        # Compute learned sensitivities using candidate as the forced driver.
        candidate_sensitivities = {}
        driver_data = filtered_hist[candidate]
        for factor, data in filtered_hist.items():
            if factor == candidate:
                candidate_sensitivities[factor] = 1.0
            else:
                candidate_sensitivities[factor] = compute_sensitivity(data, driver_data)

        # Generate future data with candidate forced to increase.
        baseline_future = generate_future_data(t_future, filtered_hist, candidate_sensitivities, main_factor=candidate, rate=0.0)
        increased_future = generate_future_data(t_future, filtered_hist, candidate_sensitivities, main_factor=candidate, rate=rate)

        baseline_filtered = filter_future_signals(baseline_future)
        increased_filtered = filter_future_signals(increased_future)

        unified_baseline = build_unified_profile(baseline_filtered)
        unified_increased = build_unified_profile(increased_filtered)
        business_profile_change = (np.mean(unified_increased) - np.mean(unified_baseline)) / np.mean(unified_baseline) * 100

        # Instead of candidate's own change, measure the change in the outcome variable.
        baseline_outcome_avg = np.mean(baseline_filtered[outcome])
        increased_outcome_avg = np.mean(increased_filtered[outcome])
        outcome_change = (increased_outcome_avg - baseline_outcome_avg) / baseline_outcome_avg * 100

        financial_impact = base_value * (outcome_change / 100)

        if financial_impact < 0:
            financial_impact = financial_impact * -1
            if financial_impact < 1000:

                result = {
                    "name": candidate,
                    outcome: outcome_change,  # This is the driver effect on the outcome.
                    "unified_business_profile": business_profile_change,
                    "financial_impact": financial_impact 
                }
                #print(result["name"], result[outcome])
                results_table.append(result)

    # Sort candidates by the outcome change (driver effect) in descending order.
    sorted_by_effect = sorted(results_table, key=lambda x: x["financial_impact"] if x["financial_impact"] < 1000 else float('inf'), reverse=True)

    #print(sorted_by_effect,"\n\n")
    
    

    # Extract top 3 strengths and bottom 3 weaknesses.
    
    weaknesses = sorted_by_effect[-8:]
    #print("\n\n", weaknesses)

    weaknesses_str = "\n".join([
        f"• **{entry['name']}** : {outcome.capitalize()} impact increased by {abs(entry[outcome]):.2f}%, with a increase of {abs(entry['unified_business_profile']):.2f}% in the business profile and a potential ROI gain of ${abs(entry['financial_impact']):.2f}.\n An ROI of less than $1000 signifies a potential loss. \n"
        for entry in weaknesses
    ])



    summary = (
        "-----------------------------\n\n"
        
        f" ## Overview:\n\n"
        f"An extensive evaluation of various business drivers was conducted to assess their impact on {outcome}. "
        f"This analysis compared changes in {outcome} performance and the overall business profile, while also estimating potential returns based on a baseline investment of $1000.\n\n"
        f" ### **Areas of Vulnerability:**\n\n"
        f"{weaknesses_str}\n\n"
        f"### **Detailed Analysis:**\n"
        
        
        f"This underperformance is linked to notable declines in {outcome} performance and overall business health, with potential ROI gains falling below the investment threshold.\n\n"
        f" ### **Recommendations:**\n"
        f"- **Address Vulnerabilities:** Focus on identifying the root causes behind the weak drivers and implement targeted strategies to mitigate losses and improve overall performance.\n\n"
        f" ### **Conclusion:**\n"
        f"By strategically enhancing the strengths and addressing the vulnerabilities, significant improvements in {outcome} performance and overall business health can be achieved. "
        f"This balanced approach is expected to drive higher financial returns, promote sustainable growth, and optimize operational efficiency over time."
    )
    return summary

