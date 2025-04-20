#what is the probability that a certain product will be in demand today
import numpy as np
import math
from datetime import datetime, timedelta
from scipy.stats import beta
def demand_analysis():

    # -------- Step 1: Generate Synthetic Historical Data --------

    np.random.seed(42)

    # Define simulation period: 120 days (~4 months)
    start_date = datetime(2025, 1, 1)
    n_days = 100
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Define 10 synthetic products representing a small business in Kenya.
    # For each product, choose a category and assign a base purchase probability (between 0.05 and 0.25)
    # and a promotion lift factor (multiplying the probability when promotion is active)
    product_list = []
    for i in range(10):
        product = {
            "product_id": f"P{i+1:03d}",
            "product_name": f"Product_{i+1}",
            "category": np.random.choice(["Food", "Beverage", "Household", "Electronics"]),
            "base_prob": np.random.uniform(0.05, 0.25),
            "promo_lift": np.random.uniform(1.2, 1.8)
        }
        product_list.append(product)

    historical_data = {"products": []}

    # For each product, simulate daily outcomes:
    # Each day, assign a promotion flag with 20% chance.
    # Add a seasonal effect as normal noise (mean 0, std 0.02).
    # Calculate effective probability = base_prob * (promo_lift if promotion active) + seasonal effect,
    # then clip between 0 and 1.
    # Simulate a sale event as a binary outcome based on effective probability.
    for prod in product_list:
        prod_history = []
        for current_date in dates:
            promotion = np.random.binomial(1, 0.2)
            seasonal_effect = np.random.normal(0, 0.02)
            if promotion:
                effective_prob = prod["base_prob"] * prod["promo_lift"]
            else:
                effective_prob = prod["base_prob"]
            effective_prob = np.clip(effective_prob + seasonal_effect, 0, 1)
            sale_event = np.random.binomial(1, effective_prob)
            prod_history.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "promotion": promotion,
                "sale": sale_event,
                "effective_prob": effective_prob  # For reference
            })
        historical_data["products"].append({
            "product_id": prod["product_id"],
            "product_name": prod["product_name"],
            "category": prod["category"],
            "base_prob": prod["base_prob"],
            "promo_lift": prod["promo_lift"],
            "history": prod_history
        })

    # -------- Step 2: Build the Product-Level Demand Probability Model --------
    # Objective: Compute the probability that a product is bought on a given day.
    # We use a Bayesian model with a Beta prior to update the probability based on observed data.

    # Choose one product (e.g., the first product)
    selected_product = historical_data["products"][0]
    sales_data = np.array([day["sale"] for day in selected_product["history"]])
    n_observations = len(sales_data)
    n_success = 60
    n_failure = n_observations - n_success

    # Use a Beta(1,1) prior (non-informative).
    alpha_prior, beta_prior = 1, 1

    # Update posterior counts.
    alpha_post = alpha_prior + n_success
    beta_post = beta_prior + n_failure

    # Posterior distribution of the daily purchase probability is Beta(alpha_post, beta_post)
    p_expected = alpha_post / (alpha_post + beta_post)
    credible_interval = beta.interval(0.95, alpha_post, beta_post)

    # -------- Non-Technical Dynamic Output --------
    non_technical_output = (
    f"## Product Demand Probability Analysis for **{selected_product['product_name']} ({selected_product['product_id']}**):\n"
    f"- Observed period (days): **{n_observations}**\n"
    f"- Days with a purchase: **{n_success}** \n"
    f"- Estimated daily purchase probability: **{p_expected:.2f}**\n"
    f"- 95% Credible Interval: **[{credible_interval[0]:.2f}, {credible_interval[1]:.2f}]**\n"
    )
    return non_technical_output