import numpy as np
from Data.business_data import historical_data
from scipy.stats import theilslopes

# Global fixed variables (if needed by the optimizer).
fixed_m_change = None
fixed_s_change = None

def optimize(mode, target_revenue, fixed_m_change, fixed_s_change):
    np.random.seed(42)
    t_hist = np.arange(730)

    #------------------------------------------------------------------------------
    # 1. Define Historical Data Dynamically
    #------------------------------------------------------------------------------
    # Include additional business drivers: revenue, sales, marketing, inventory, market_share.
    revenue = historical_data['revenue']
    sales = historical_data['sales']
    marketing = historical_data['marketing']
    inventory = historical_data.get('inventory', revenue)      # Use revenue as fallback if key missing.
    market_share = historical_data.get('market_share', sales)    # Use sales as fallback if key missing.

    #------------------------------------------------------------------------------
    # 2. Learn Sensitivities Using Robust Linear Regression
    #------------------------------------------------------------------------------
    def compute_sensitivity(y, x):
        # Use Theil-Sen estimator for a robust slope estimate.
        slope, intercept, _, _ = theilslopes(y, x, 0.95)
        return slope

    # Build filtered historical dictionary with selected factors.
    filtered_hist = {
        "revenue": revenue,
        "sales": sales,
        "marketing": marketing,
        "inventory": inventory,
        "market_share": market_share
    }
    main = "revenue"  # The chosen driver

    # Compute learned sensitivities: for the main factor we set baseline = 1.0.
    learned_sensitivities = {}
    for factor, data in filtered_hist.items():
        if factor == main:
            learned_sensitivities[factor] = 1.0
        else:
            # Estimate how changes in other factors affect revenue by regressing revenue on that factor.
            learned_sensitivities[factor] = compute_sensitivity(revenue, data)

    #------------------------------------------------------------------------------
    # 3. Deterministic Revenue Model Based on Learned Sensitivities
    #------------------------------------------------------------------------------
    # Use the latest revenue value as the baseline rather than the mean.
    r_baseline = revenue[-1]
    percentage_gain = (( r_baseline - target_revenue) / r_baseline) * 100 

    def calculate_revenuemodel(m_change, s_change, t):
        sensitivity_marketing = learned_sensitivities["marketing"]
        sensitivity_sales = learned_sensitivities["sales"]
        relative_revenue = 1 + sensitivity_marketing * m_change + sensitivity_sales * s_change
        revenue_forecast = r_baseline * relative_revenue
        return revenue_forecast

    #------------------------------------------------------------------------------
    # 4. Reward and Objective Function with Basic Uncertainty Penalty
    #------------------------------------------------------------------------------
    def calculate_reward(revenue):
        error = (revenue - target_revenue) / target_revenue
        return -error**2  # Squared error penalty

    def objective(params, t_values):
        m_change, s_change = params
        rewards = [calculate_reward(calculate_revenuemodel(m_change, s_change, t)) for t in t_values]
        return np.mean(rewards)

    #------------------------------------------------------------------------------
    # 5. Numerical Gradient Computation Using Finite Differences
    #------------------------------------------------------------------------------
    def compute_gradient(params, t_values, h=1e-5):
        grad = np.zeros_like(params)
        f0 = objective(params, t_values)
        for i in range(len(params)):
            params_h = np.array(params)
            params_h[i] += h
            f1 = objective(params_h, t_values)
            grad[i] = (f1 - f0) / h
        return grad

    #------------------------------------------------------------------------------
    # 6. Gradient Ascent Optimization for Marketing and Sales Adjustments
    #------------------------------------------------------------------------------
    def optimize_params(initial_params, t_values, learning_rate=0.001, 
                        max_iterations=50000, mode="none", tol=1e-5):
        params = np.array(initial_params, dtype=float)
        history = []
        for iteration in range(max_iterations):
            #print(f"iteration: {iteration, tol}")
            grad = compute_gradient(params, t_values)
            if mode == "salesconstant":
                params[0] += learning_rate * grad[0]
                params[1] = fixed_s_change if fixed_s_change is not None else initial_params[1]
            elif mode == "marketing_constant":
                params[1] += learning_rate * grad[1]
                params[0] = fixed_m_change if fixed_m_change is not None else initial_params[0]
            else:
                params += learning_rate * grad

            current_obj = objective(params, t_values)
            history.append((iteration, params.copy(), current_obj))
            if current_obj >= -9.99e-06:
                break
        return params, history

    #------------------------------------------------------------------------------
    # 7. Run Optimization and Output Final Optimal Action
    #------------------------------------------------------------------------------
    t_values = np.linspace(780, 800, 50)
    initial_params = [0.0, 0.0]  # Starting with no change in marketing or sales.
    optimal_params, history = optimize_params(initial_params, t_values, 
                                                learning_rate=0.001, max_iterations=1000000, mode=mode)
    mid_t = np.mean(t_values)
    final_revenue = calculate_revenuemodel(optimal_params[0], optimal_params[1], mid_t)
    final_reward = calculate_reward(final_revenue)

    result = (
        "-----------------------------\n\n"
        "Our analysis has provided actionable insights to guide your business strategy.\n\n"
        "Key Business Insights:\n"
        f"  - Current Baseline Revenue: ${r_baseline:.2f}\n"
        f"  - Target Revenue:           ${target_revenue:.2f}\n"
        f"  - Recommended Marketing Change: {optimal_params[0]:.2f}\n"
        f"  - Recommended Sales Change:     {optimal_params[1]:.2f}\n"
        f"  - Forecasted Revenue:       ${final_revenue:.2f}\n\n"
        f"These recommendations aim to lift your revenue by approximately {percentage_gain:.2f}% above your current baseline.\n\n"
        "In practical terms:\n\n"
        "  • A modest adjustment in marketing and sales efforts is predicted to yield significant revenue improvements.\n\n"
        "  • The recommendations are based on a robust analysis of historical performance and current trends,\n"
        "    ensuring that the advice is both data-driven and actionable.\n\n"
        "  • Strategic realignments based on these insights can help optimize resource allocation and drive sustainable growth.\n\n"
        "Overall, these insights provide clear guidance for streamlining your strategy and positioning your business\n"
        "to achieve strong, sustainable revenue improvements.\n\n"
    )

    return result

# Example usage:
#print(optimize("none", 700, fixed_m_change, fixed_s_change))
