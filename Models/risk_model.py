import numpy as np
from collections import defaultdict
from datetime import datetime
from Models.corrective_assessment_support_for_risk_model import corrective  # Import the unified corrective function
from Data.business_data import historical_data

# --- 1. Improved Parameter Calibration Using Historical Data and Bayesian Update ---
def calibrate_params(historical_data, kpi_names, factor, decay_rate=0.03):
    """
    Calibrates model parameters based on historical data.
    Uses exponential weighting to favor recent data and a Bayesian-inspired update.
    """
    params = {}
    for kpi in kpi_names:
        data = np.array(historical_data[kpi])
        n = len(data)
        # Compute exponential weights for each historical observation (most recent gets the highest weight)
        weights = np.exp(np.linspace(-decay_rate * (n-1), 0, n))
        weights /= np.sum(weights)
        weighted_std = np.sqrt(np.sum(weights * (data - np.sum(weights * data)) ** 2))
        sigma = weighted_std
        
        # Customized parameter values for the prioritized factor
        alpha = 1.8 if kpi == factor else 1.2
        beta = 2.5 if kpi == factor else 1.8
        
        # Base estimate for k based on sigma and sensitivity constant
        k_obs = 0.6 + sigma * 0.01
        
        # Bayesian update: blend a fixed prior with the observed value
        prior_k = 0.6
        prior_weight = 5  # effective prior sample size
        observed_weight = n
        k_updated = (prior_weight * prior_k + observed_weight * k_obs) / (prior_weight + observed_weight)
        
        gamma = 1.0
        omega = 0.08 if kpi == factor else 0.1

        params[kpi] = {'alpha': alpha, 'beta': beta, 'k': k_updated, 'gamma': gamma, 'omega': omega}
    return params

# --- 2. Monitoring and Diagnostics Functions ---
def record_deviation(health_log, kpi, magnitude, duration, time):
    health_log['historical_deviations'][kpi].append((magnitude, duration, time))
    health_log['diagnostic_log'].append(f"Deviation for {kpi}: magnitude {magnitude:.4f} at time {time}")

def update_action_effectiveness(health_log, action, effectiveness):
    health_log['action_effectiveness'][action].append(effectiveness)
    health_log['diagnostic_log'].append(f"Action '{action}' effectiveness updated to {effectiveness:.4f}")

def get_actions(health_log, kpi, current_state, trend, x_eq):
    # Compare current state to equilibrium target x_eq (assumed to be a one-element array)
    deviation = x_eq[0] - current_state
    health_log['diagnostic_log'].append(
        f"{kpi}: current_state={current_state:.4f}, target={x_eq[0]:.4f}, deviation={deviation:.4f}"
    )
    if deviation < 0:
        return ['System stable']
    return [corrective(kpi)]


# --- 3. Controlled Chaos Model (Core Simulation Function) with Stochastic Intervention ---
memory_factor = 0.8
def controlled_chaos_model(x, t, kpi, params, chaotic_inputs, prev_dx, x_eq, health_log):
    """
    Computes the state change (dx) for the KPI.
    Combines chaotic inputs, inertia, corrective force, external cyclical correction, and stochastic intervention.
    """
    par = params[kpi]
    alpha = par['alpha']
    beta = par['beta']
    k_val = par['k']
    gamma = par['gamma']
    omega = par['omega']

    # Retrieve historical deviations; if none, use default zero
    historical_devs = health_log['historical_deviations'].get(kpi, [(0, 0, 0)])
    historical_avg = np.mean([d[0] for d in historical_devs])
    if historical_avg > 0.1 * x_eq[0]:
        alpha *= 0.9

    f_chaos = chaotic_inputs[kpi][t]
    inertia = memory_factor * prev_dx
    dynamic_change = alpha * f_chaos * np.exp(-beta * abs(x - x_eq[0]))
    corrective_force = -k_val * (x - x_eq[0])
    external_correction = gamma * np.sin(omega * t)
    
    # Stochastic intervention: with 30% probability apply a random fraction of the corrective force.
    intervention = 0
    if (x < x_eq[0]) and ((x_eq[0] - x) > 0.1 * x_eq[0]):
        if np.random.rand() < 0.3:
            intervention = -np.random.uniform(0.05, 0.15) * corrective_force
    dx = dynamic_change + corrective_force + external_correction + inertia + intervention

    return dx

# --- 4 & 6. run_risk_analysis Function with Robust State Handling and Business Integration ---
def run_risk_analysis(factor):
    """
    Executes a Monte Carlo simulation and risk analysis for the given KPI.
    Incorporates calibrated parameters, a controlled chaos simulation, and detailed diagnostics.
    
    Returns a dictionary with a markdown narrative, visualization data, and diagnostic logs.
    """
    if factor not in historical_data:
        raise ValueError(f"Factor '{factor}' not found in historical data.")
    
    kpi_names = [factor]
    data = historical_data[factor]
    
    # Set chaotic inputs to the historical data
    chaotic_inputs = {factor: data}
    improve = 1.10  # Factor to calculate equilibrium target
    x_eq = np.array([data[-1] * improve])
    
    # Pre-calculate corrective recommendation (for recommended actions)
    rec = corrective(factor)
    
    # Calibrate parameters with improved Bayesian updating and exponential weighting
    params = calibrate_params(historical_data, kpi_names, factor)
    
    # Set up health state as a dictionary
    health_log = {
        'historical_deviations': defaultdict(list),
        'action_effectiveness': defaultdict(list),
        'diagnostic_log': []
    }
    
    # Simulation Setup
    time_steps = 730  # Two years of simulation (days)
    dt = 1
    num_simulations = 10
    np.random.seed(42)
    prev_dx = {factor: 0}
    
    simulations = {factor: np.zeros((num_simulations, time_steps))}
    days_to_target = []
    tolerance = 0.30 * x_eq[0]
    
    # Run Monte Carlo simulation for each simulation run
    for sim in range(num_simulations):
        sim_state = {factor: data[-1]}
        critical_state = {factor: None}
        target_reached = False
        prev_dx = {factor: 0}
        
        for t in range(time_steps):
            dx = controlled_chaos_model(sim_state[factor], t, factor, params, chaotic_inputs,
                                          prev_dx[factor], x_eq, health_log)
            prev_dx[factor] = dx
            sim_state[factor] += dx * dt
            simulations[factor][sim, t] = sim_state[factor]
            current_eq = x_eq[0]
            deviation = sim_state[factor] - current_eq
            trend = 'down' if deviation < 0 else 'up'
            record_deviation(health_log, factor, abs(deviation), dt, t)
            
            # Detect a critical state when deviation drops below a threshold
            if deviation < -0.2 * current_eq:
                if critical_state[factor] is None:
                    critical_state[factor] = t
            else:
                if critical_state[factor] is not None and deviation > -0.1 * current_eq:
                    recovery_time = t - critical_state[factor]
                    health_log['recovery_patterns'] = health_log.get('recovery_patterns', defaultdict(list))
                    health_log['recovery_patterns'][factor].append((recovery_time, abs(deviation)))
                    # Update corrective action effectiveness for each recommended action
                    for action in get_actions(health_log, factor, sim_state[factor], trend, x_eq):
                        effectiveness = 1 / recovery_time if recovery_time > 0 else 0
                        update_action_effectiveness(health_log, action, effectiveness)
                    critical_state[factor] = None
            
            # Record when the simulation reaches within tolerance of equilibrium
            if not target_reached and abs(sim_state[factor] - current_eq) <= tolerance:
                days_to_target.append(t)
                target_reached = True
    
    if len(days_to_target) < num_simulations:
        days_to_target.extend([np.nan] * (num_simulations - len(days_to_target)))
    
    # Calculate weighted recovery time from logged recovery patterns
    if 'recovery_patterns' in health_log and health_log['recovery_patterns'].get(factor):
        rec_data = health_log['recovery_patterns'][factor]
        total_weight = sum(w for (_, w) in rec_data)
        weighted_recovery = sum(t * w for (t, w) in rec_data) / total_weight if total_weight > 0 else np.nan
    else:
        weighted_recovery = np.nan

    # Factor-specific historical growth rate calculation
    if len(data) >= 31:
        recent_growth_rates = np.diff(data[-365:])
        historical_growth_rate = np.mean(recent_growth_rates)
    else:
        historical_growth_rate = np.mean(np.diff(data))
        
    if historical_growth_rate > 0:
        expected_recovery_days_historical = (x_eq[0] - data[-1]) / historical_growth_rate
    else:
        expected_recovery_days_historical = np.nan

    sim_target_avg = np.nanmean(days_to_target)
    if not np.isnan(expected_recovery_days_historical) and not np.isnan(sim_target_avg):
        avg_target_days = (sim_target_avg + expected_recovery_days_historical) / 2.0
    elif not np.isnan(expected_recovery_days_historical):
        avg_target_days = expected_recovery_days_historical
    else:
        avg_target_days = sim_target_avg

    realistic_recovery_time = max(weighted_recovery, expected_recovery_days_historical)
    
    # --- 6. Build Detailed Markdown Summary for the Factor ---
    current_value = data[-1]
    target_value = x_eq[0]
    deviation = abs(target_value - current_value)
    trend = "down" if current_value < target_value else "up"
    risk_level = ("Severe" if current_value < 0.7 * target_value 
                  else "Moderate" if current_value < 0.9 * target_value 
                  else "Low")
    actions = ", ".join(get_actions(health_log, factor, current_value, trend, x_eq))
    expected_recovery = f"{realistic_recovery_time:.1f} days" if not np.isnan(realistic_recovery_time) else "N/A"

    summary_lines = []
    summary_lines.append("-----------------------------\n")
    summary_lines.append("# Risk Report\n")
    summary_lines.append(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    summary_lines.append("### Overview\n")
    summary_lines.append(
        f"This report details the current risk status and recovery expectations for the KPI: **{factor.capitalize()}**. "
        f"Using Bayesian-updated parameters and a controlled chaos simulation with stochastic interventions, the model provides actionable insights."
    )
    summary_lines.append("\n### KPI Status Summary\n")
    summary_lines.append(f"#### {factor.capitalize()} Status")
    summary_lines.append(f"- **Target:** {target_value:.1f}")
    summary_lines.append(f"- **Current Value:** {current_value:.1f}")
    summary_lines.append(f"- **Deviation:** {deviation:.1f} units")
    summary_lines.append(f"- **Trend:** {trend}")
    summary_lines.append(f"- **Risk Level:** {risk_level}")
    summary_lines.append(f"- **Recommended Actions:** {actions}")
    summary_lines.append(f"- **Expected Recovery Time:** {expected_recovery}\n")
    summary_lines.append("-----------------------------")
    risk_summary = "\n".join(summary_lines)

    # Build risk visualization data (for integration into business planning)
    risk_visualization = {
        factor + "_status": {
            "target": target_value,
            "current_value": current_value,
            "risk_level": risk_level,
            "expected_recovery_time": realistic_recovery_time,
            "recommended_actions": get_actions(health_log, factor, current_value, trend, x_eq)
        }
    }

    risk_data = {
        "narrative": risk_summary,
        "visual": risk_visualization,
        "diagnostics": health_log['diagnostic_log']
    }
    return risk_data

# Example usage:
#risk_report = run_risk_analysis("revenue")
#print(risk_report["narrative"])
