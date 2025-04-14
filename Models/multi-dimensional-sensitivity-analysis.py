import numpy as np
from Data.business_data import historical_data


# -------------------------------
# Higher-Order Sensitivity Analysis (Interaction-Only)
# -------------------------------
def learn_higher_order_sensitivities(historical_data, target_factor):
    """
    Learns first-order and interaction sensitivities of the target factor
    with respect to all other factors using linear regression.
    Does not include quadratic terms.
    """
    y = np.array(historical_data[target_factor])
    base_factors = [f for f in historical_data.keys() if f != target_factor]
    base_data = [historical_data[f] for f in base_factors]
    X_base = np.column_stack(base_data)

    # Standardize predictors
    X_mean = np.mean(X_base, axis=0)
    X_std = np.std(X_base, axis=0)
    X_std[X_std == 0] = 1.0
    X_normalized = (X_base - X_mean) / X_std

    # Build interaction terms (no quadratic terms)
    interaction_terms = []
    interaction_names = []
    n = X_normalized.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            interaction_terms.append(X_normalized[:, i] * X_normalized[:, j])
            interaction_names.append(f"{base_factors[i]} * {base_factors[j]}")

    X_full = np.column_stack([X_normalized] + interaction_terms)
    factor_names = base_factors + interaction_names

    # Add intercept
    X_design = np.column_stack([np.ones(len(y)), X_full])

    # Regression
    beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
    intercept = beta[0]

    # Residual variance and SE estimation
    dof = len(y) - rank
    residual_var = residuals[0] / dof if dof > 0 else 0.0
    cov_beta = residual_var * np.linalg.inv(X_design.T @ X_design)

    # Extract sensitivities and SEs
    sensitivities = {}
    se = {}
    for i, name in enumerate(factor_names):
        sensitivities[name] = beta[i + 1]
        se[name] = np.sqrt(cov_beta[i + 1, i + 1])

    return intercept, sensitivities, se


# -------------------------------
# Unified Corrective Recommendation Function
# -------------------------------
def corrective(target_factor):
    """
    Generates corrective recommendations using higher-order (interaction-based) sensitivity analysis.
    """
    intercept, sensitivities, se = learn_higher_order_sensitivities(historical_data, target_factor)

    # Filter for positive-impact terms
    positive_factors = {f: sensitivities[f] for f in sensitivities if sensitivities[f] > 0}
    if not positive_factors:
        return f"No actionable sensitivities found for {target_factor}."

    # Sort and select top 5
    sorted_factors = sorted(positive_factors.items(), key=lambda x: x[1], reverse=True)
    top_five = sorted_factors[:5]

    # Build recommendations
    recommendations = []
    for factor, coeff in top_five:
        recommendations.append(
            f"\nIncrease {factor} efforts **(sensitivity: {coeff:.4f} ± {se[factor]:.4f})**.\n\n"
        )

    rec_message = (
        f"\n\nCorrective recommendations for improving {target_factor.capitalize()}:\n" +
        "\n".join(recommendations)
    )
    return rec_message
print(corrective("revenue"))