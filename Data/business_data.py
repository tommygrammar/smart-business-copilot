import numpy as np
np.random.seed(42)
t_hist = np.arange(730)  # Historical period

historical_data = {
    "sales": (100 + 0.5 * t_hist + 10 * np.sin(2 * np.pi * t_hist / 7) + np.random.normal(0, 2, t_hist.shape)),
    "revenue": 150 + 0.6 * t_hist + 8 * np.sin(2 * np.pi * (t_hist + 2) / 7) + np.random.normal(0, 2, t_hist.shape),
    "marketing": 50 + 0.2 * t_hist + 3 * np.sin(2 * np.pi * (t_hist + 3) / 7) + np.random.normal(0, 1, t_hist.shape),
    "customer_sat": 80 + 0.1 * t_hist + 5 * np.sin(2 * np.pi * (t_hist + 1) / 7) + np.random.normal(0, 1.5, t_hist.shape),
    "website_traffic": 300 + 1.0 * t_hist + 20 * np.sin(2 * np.pi * (t_hist + 4) / 7) + np.random.normal(0, 5, t_hist.shape),
    "employee_productivity": 70 + 0.3 * t_hist + 4 * np.sin(2 * np.pi * (t_hist + 2) / 7) + np.random.normal(0, 1, t_hist.shape),
    "operational_efficiency": 90 + 0.4 * t_hist + 6 * np.sin(2 * np.pi * (t_hist + 5) / 7) + np.random.normal(0, 2, t_hist.shape),
    "competitive_advantage": 100 + 0.2 * t_hist + 7 * np.sin(2 * np.pi * (t_hist + 3) / 7) + np.random.normal(0, 2, t_hist.shape),
    "inventory": 200 + 0.3 * t_hist + 5 * np.sin(2 * np.pi * t_hist / 7) + np.random.normal(0, 2, t_hist.shape),
    # Additional Factors...
    "customer_loyalty": 70 + 0.2 * t_hist + 4 * np.sin(2 * np.pi * (t_hist + 1) / 10) + np.random.normal(0, 1.5, t_hist.shape),
    "brand_awareness": 80 + 0.25 * t_hist + 6 * np.sin(2 * np.pi * (t_hist + 2) / 10) + np.random.normal(0, 1.5, t_hist.shape),
    "cost_efficiency": 90 + 0.3 * t_hist + 5 * np.sin(2 * np.pi * (t_hist + 3) / 10) + np.random.normal(0, 2, t_hist.shape),
    "supply_chain_reliability": 85 + 0.15 * t_hist + 5 * np.sin(2 * np.pi * (t_hist + 4) / 10) + np.random.normal(0, 2, t_hist.shape),
    "innovation": 60 + 0.4 * t_hist + 7 * np.sin(2 * np.pi * (t_hist + 5) / 10) + np.random.normal(0, 2, t_hist.shape),
    "employee_satisfaction": 75 + 0.2 * t_hist + 3 * np.sin(2 * np.pi * (t_hist + 6) / 10) + np.random.normal(0, 1.5, t_hist.shape),
    "market_share": 30 + 0.1 * t_hist + 2 * np.sin(2 * np.pi * (t_hist + 7) / 10) + np.random.normal(0, 1, t_hist.shape),
    "digital_engagement": 200 + 1.2 * t_hist + 15 * np.sin(2 * np.pi * (t_hist + 8) / 10) + np.random.normal(0, 3, t_hist.shape),
    "social_media_presence": 50 + 0.8 * t_hist + 8 * np.sin(2 * np.pi * (t_hist + 9) / 10) + np.random.normal(0, 1.5, t_hist.shape),
    "product_quality": 95 + 0.05 * t_hist + 4 * np.sin(2 * np.pi * (t_hist + 10) / 10) + np.random.normal(0, 1, t_hist.shape),
    "operational_costs": 500 + 0.7 * t_hist + 20 * np.sin(2 * np.pi * (t_hist + 11) / 10) + np.random.normal(0, 5, t_hist.shape),
    "financial_health": 100 + 0.5 * t_hist + 10 * np.sin(2 * np.pi * (t_hist + 12) / 10) + np.random.normal(0, 2, t_hist.shape)
}
