#what are my weakest and strongest products
import numpy as np
import math
from datetime import datetime, timedelta
from scipy.stats import beta


# -------------------------------
# Step 1: Generate Synthetic Data
# -------------------------------

def type_shit():

    np.random.seed(42)

    # Assume our business has 10 products.
    n_products = 10

    # For simplicity, we use synthetic metrics that might vary across products:
    # - Sales velocity: average units sold per day (normally distributed around 20 with std dev 5).
    # - Purchase frequency: average purchases per week (around 15 with std dev 4).
    # - Customer review score: average rating (scale of 1-5, normalized to 0-1).
    # - Margin: profit margin percentage (randomly between 10% and 50%, normalized to 0-1).
    #
    # We also add a small noise term to mimic natural variability.

    product_metrics = []
    for i in range(n_products):
        sales_velocity = np.random.normal(20, 5)  # units per day
        purchase_freq = np.random.normal(15, 4)   # purchases per week
        customer_review = np.random.normal(4, 0.5)  # rating between 1 and 5
        margin = np.random.uniform(0.10, 0.50)      # as a fraction
        product_metrics.append({
            "product_id": f"P{i+1:03d}",
            "product_name": f"Product_{i+1}",
            "category": np.random.choice(["Food", "Beverage", "Household", "Electronics"]),
            "sales_velocity": sales_velocity,
            "purchase_freq": purchase_freq,
            "customer_review": np.clip(customer_review, 1, 5) / 5,  # normalized to 0-1
            "margin": margin
        })

    # -------------------------------
    # Step 2: Performance Scoring using Probabilistic Clustering
    # -------------------------------
    #
    # We combine the metrics into a composite performance score.
    # For each metric, we compute a z-score (standard score) then combine them with weights.
    # We add a small noise term to simulate measurement uncertainty.

    # Extract raw metric values
    sales_velocities = np.array([p["sales_velocity"] for p in product_metrics])
    purchase_freqs = np.array([p["purchase_freq"] for p in product_metrics])
    customer_reviews = np.array([p["customer_review"] for p in product_metrics])
    margins = np.array([p["margin"] for p in product_metrics])

    # Standardize each metric
    def z_score(arr):
        return (arr - np.mean(arr)) / np.std(arr)

    z_sales = z_score(sales_velocities)
    z_purchase = z_score(purchase_freqs)
    z_reviews = z_score(customer_reviews)
    z_margin = z_score(margins)

    # Define weights (importance of each metric)
    w_sales = 0.35
    w_purchase = 0.25
    w_reviews = 0.20
    w_margin = 0.20

    # Calculate the composite performance score for each product
    sigma_noise = 0.1
    performance_scores = (w_sales * z_sales +
                        w_purchase * z_purchase +
                        w_reviews * z_reviews +
                        w_margin * z_margin +
                        np.random.normal(0, sigma_noise, n_products))

    # Attach the composite performance score to each product record
    for i, prod in enumerate(product_metrics):
        prod["performance_score"] = performance_scores[i]

    # -------------------------------
    # Step 3: Segment Products into Strong, Mid, Weak Tiers
    # -------------------------------
    #
    # Use percentile thresholds to assign each product a performance tier.
    strong_threshold = np.percentile(performance_scores, 67)  # top 33%
    weak_threshold = np.percentile(performance_scores, 33)      # bottom 33%

    for prod in product_metrics:
        score = prod["performance_score"]
        if score >= strong_threshold:
            prod["tier"] = "Strong"
        elif score <= weak_threshold:
            prod["tier"] = "Weak"
        else:
            prod["tier"] = "Mid"

    # -------------------------------
    # Step 4: Prepare and Print a Comprehensive Output
    # -------------------------------
    #
    # The output includes product details and explicitly lists which products are Strong, Mid, and Weak.

    output_lines = []
    output_lines.append("Product Performance Segmentation Analysis:\n")
    output_lines.append("Metrics: Sales Velocity (units/day), Purchase Frequency (purchases/week), Customer Review (norm), Margin.\n")
    output_lines.append("A composite performance score has been computed, categorizing products as Strong, Mid, or Weak.\n")
    output_lines.append(f"Strong threshold (>= {strong_threshold:.2f}); Weak threshold (<= {weak_threshold:.2f}).\n\n")

    # Collect lists for each tier for summary
    strong_products = []
    mid_products = []
    weak_products = []
    output_lines.append("--------------------------------------------------------------------------------\n")

    for prod in product_metrics:
        category = prod.get("category", "N/A")

        output_lines.append(f"- Product ID: {prod['product_id']} | Name: {prod['product_name']} | Category: {category}\n")
        output_lines.append(f"  Sales Velocity: {prod['sales_velocity']:.2f} units/day  |  Purchase Frequency: {prod['purchase_freq']:.2f} purchases/week\n")
        output_lines.append(f"  Customer Review (norm): {prod['customer_review']:.2f}  |  Margin: {prod['margin']:.2f}\n")
        output_lines.append(f"  Composite Performance Score: {prod['performance_score']:.2f}  =>  Tier: {prod['tier']}\n\n\n")

        
        if prod["tier"] == "Strong":
            strong_products.append(prod["product_name"])
        elif prod["tier"] == "Mid":
            mid_products.append(prod["product_name"])
        else:
            weak_products.append(prod["product_name"])

    output_lines.append("\nSummary of Segmentation:\n")
    output_lines.append(f"Strong Products: {', '.join(strong_products)}\n")
    output_lines.append(f"Mid Products: {', '.join(mid_products)}\n")
    output_lines.append(f"Weak Products: {', '.join(weak_products)}\n")

    output_summary = "".join(output_lines)
    #print(output_summary)
    return output_summary
