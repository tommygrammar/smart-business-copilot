import numpy as np


days = 730  # 2 years of daily data

np.random.seed(42)

# Base starting values for each factor (roughly in realistic ranges)
base_values = {
    "sales": 200,
    "revenue": 150,
    "marketing": 70,
    "customer_sat": 80,
    "website_traffic": 300,
    "employee_productivity": 75,
    "operational_efficiency": 85,
    "competitive_advantage": 90,
    "inventory": 250,
    "customer_loyalty": 70,
    "brand_awareness": 75,
    "cost_efficiency": 80,
    "supply_chain_reliability": 80,
    "innovation": 60,
    "employee_satisfaction": 70,
    "market_share": 30,
    "digital_engagement": 200,
    "social_media_presence": 60,
    "product_quality": 90,
    "operational_costs": 600,      # high costs
    "financial_health": 80,
    "rnd_expenses": 50,
    "customer_acquisition_cost": 30,
    "net_profit": 20,              # low profit base (will turn negative)
    "employee_turnover": 15,
    "cybersecurity_incidents": 5,
    "regulatory_compliance": 90,
    "sustainability_index": 60,
    "corporate_social_responsibility": 70,
    "customer_referral_rate": 15,
    "operational_risk": 10,
    "innovation_index": 55,
    "market_volatility": 25,
    "investment_in_infrastructure": 80,
    "workforce_diversity": 50,
    "patent_activity": 5,
    "digital_transformation_index": 65
}

# For a loss-making business, we'll impose:
# - Low revenue growth (or even decline)
# - High and erratic operational costs
# - Net profit that declines (or remains negative)

# We'll create an irregular random walk for each factor:
def generate_series(base, drift=0, volatility=1.0, days=730):
    # Generate a series as a cumulative sum of random fluctuations plus a small drift.
    fluctuations = np.random.normal(loc=drift, scale=volatility, size=days)
    series = base + np.cumsum(fluctuations)
    # Clip the series to keep values within realistic bounds
    series = np.clip(series, a_min=0, a_max=None)
    return series

t_hist = np.arange(days)

historical_data = {
    "sales": generate_series(base_values["sales"], drift=0.05, volatility=3.0, days=days),
    # Revenue grows slowly (or even declines) while costs eat into it
    "revenue": generate_series(base_values["revenue"], drift=-0.02, volatility=2.5, days=days),
    "marketing": generate_series(base_values["marketing"], drift=0.03, volatility=1.5, days=days),
    "customer_sat": generate_series(base_values["customer_sat"], drift=0.01, volatility=1.0, days=days),
    "website_traffic": generate_series(base_values["website_traffic"], drift=0.5, volatility=10.0, days=days),
    "employee_productivity": generate_series(base_values["employee_productivity"], drift=0.02, volatility=1.2, days=days),
    "operational_efficiency": generate_series(base_values["operational_efficiency"], drift=-0.03, volatility=1.5, days=days),
    "competitive_advantage": generate_series(base_values["competitive_advantage"], drift=0.01, volatility=2.0, days=days),
    "inventory": generate_series(base_values["inventory"], drift=0.1, volatility=4.0, days=days),
    "customer_loyalty": generate_series(base_values["customer_loyalty"], drift=0.0, volatility=1.5, days=days),
    "brand_awareness": generate_series(base_values["brand_awareness"], drift=0.02, volatility=1.5, days=days),
    "cost_efficiency": generate_series(base_values["cost_efficiency"], drift=-0.05, volatility=2.0, days=days),
    "supply_chain_reliability": generate_series(base_values["supply_chain_reliability"], drift=-0.01, volatility=2.0, days=days),
    "innovation": generate_series(base_values["innovation"], drift=0.03, volatility=2.0, days=days),
    "employee_satisfaction": generate_series(base_values["employee_satisfaction"], drift=-0.02, volatility=1.0, days=days),
    "market_share": generate_series(base_values["market_share"], drift=0.0, volatility=0.5, days=days),
    "digital_engagement": generate_series(base_values["digital_engagement"], drift=0.4, volatility=5.0, days=days),
    "social_media_presence": generate_series(base_values["social_media_presence"], drift=0.02, volatility=1.5, days=days),
    "product_quality": generate_series(base_values["product_quality"], drift=0.0, volatility=1.0, days=days),
    "operational_costs": generate_series(base_values["operational_costs"], drift=0.1, volatility=10.0, days=days),
    "financial_health": generate_series(base_values["financial_health"], drift=-0.05, volatility=2.0, days=days),
    
    "rnd_expenses": generate_series(base_values["rnd_expenses"], drift=0.05, volatility=1.5, days=days),
    "customer_acquisition_cost": generate_series(base_values["customer_acquisition_cost"], drift=0.02, volatility=1.0, days=days),
    # Net profit: revenue minus expenses; here we simulate it as a random walk that trends downward
    "net_profit": generate_series(base_values["net_profit"], drift=-0.2, volatility=3.0, days=days),
    "employee_turnover": generate_series(base_values["employee_turnover"], drift=0.01, volatility=0.5, days=days),
    "cybersecurity_incidents": generate_series(base_values["cybersecurity_incidents"], drift=0.0, volatility=0.2, days=days),
    "regulatory_compliance": generate_series(base_values["regulatory_compliance"], drift=0.0, volatility=1.0, days=days),
    "sustainability_index": generate_series(base_values["sustainability_index"], drift=-0.05, volatility=1.5, days=days),
    "corporate_social_responsibility": generate_series(base_values["corporate_social_responsibility"], drift=-0.03, volatility=1.0, days=days),
    "customer_referral_rate": generate_series(base_values["customer_referral_rate"], drift=0.0, volatility=0.5, days=days),
    "operational_risk": generate_series(base_values["operational_risk"], drift=0.0, volatility=0.3, days=days),
    "innovation_index": generate_series(base_values["innovation_index"], drift=0.02, volatility=1.5, days=days),
    "market_volatility": generate_series(base_values["market_volatility"], drift=0.0, volatility=0.8, days=days),
    "investment_in_infrastructure": generate_series(base_values["investment_in_infrastructure"], drift=0.03, volatility=1.0, days=days),
    "workforce_diversity": generate_series(base_values["workforce_diversity"], drift=0.0, volatility=0.5, days=days),
    "patent_activity": generate_series(base_values["patent_activity"], drift=0.01, volatility=0.3, days=days),
    "digital_transformation_index": generate_series(base_values["digital_transformation_index"], drift=0.02, volatility=1.5, days=days)
}


