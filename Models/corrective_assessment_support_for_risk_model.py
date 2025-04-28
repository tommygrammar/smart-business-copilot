import numpy as np
from Data.business_data import historical_data
# -------------------------------
# Sensitivity Analysis via Linear Regression
# -------------------------------
def learn_sensitivities(target_factor):
    """
    Learns the sensitivities of the target factor with respect to all other factors using
    linear regression. This version standardizes predictors for a fair comparison and computes
    rough standard errors for the coefficient estimates.
    """    
    
    # Extract response vector and predictors.
    y = np.array(historical_data[target_factor])
    factors = [f for f in historical_data.keys() if f != target_factor]
    X = np.column_stack([historical_data[f] for f in factors])
    
    # Standardize predictors.
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # Prevent division by zero.
    X_std[X_std == 0] = 1.0
    X_normalized = (X - X_mean) / X_std
    
    # Add intercept.
    X_design = np.column_stack([np.ones(len(y)), X_normalized])
    
    # Compute least squares solution.
    beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
    intercept = beta[0]
    
    # Estimate variance of residuals.
    dof = len(y) - rank  # degrees of freedom
    residual_var = residuals[0] / dof if dof > 0 else 0.0
    # Compute covariance matrix (approximate).
    cov_beta = residual_var * np.linalg.inv(X_design.T @ X_design)
    
    # Extract sensitivities for each factor (adjust for standardization) with standard errors.
    sensitivities = {}
    se = {}
    for i, factor in enumerate(factors):
        # Adjust coefficient by dividing by std of the predictor.
        coeff = beta[i + 1] / X_std[i]
        sensitivity_se = np.sqrt(cov_beta[i + 1, i + 1]) / X_std[i]
        sensitivities[factor] = coeff
        se[factor] = sensitivity_se
    return intercept, sensitivities, se

# -------------------------------
# Unified Corrective Recommendation Function
# -------------------------------
def corrective(target_factor):
    """
    Generates corrective recommendations for the specified target factor by:
      1. Learning the sensitivities (regression coefficients) and their uncertainties for the target factor
         relative to all other factors in historical_data.
      2. Filtering for factors with a positive impact.
      3. Sorting and selecting the top 5 factors based on sensitivity magnitude.
      4. Outputting user-friendly recommendations, including an indication of confidence.
    """
    intercept, sensitivities, se = learn_sensitivities(target_factor)
    
    # Filter for factors with a positive impact on the target factor.
    positive_factors = {f: sensitivities[f] for f in sensitivities if sensitivities[f] > 0}
    if not positive_factors:
        return f"No actionable sensitivities found for {target_factor}."
    
    # Sort factors by sensitivity (highest first) and take the top 5.
    sorted_factors = sorted(positive_factors.items(), key=lambda x: x[1], reverse=True)
    top_five = sorted_factors[:5]
    
    # Build recommendations with sensitivity and its estimated uncertainty.
    recommendations = []
    for factor, coeff in top_five:
        recommendations.append(
            f"\n•Increase {factor} efforts **(sensitivity: {coeff:.4f} ± {se[factor]:.4f})**.\n\n"
        )

    rec_message = (
        f"\n\nCorrective recommendations for improving {target_factor.capitalize()}:\n" +
        "\n".join(recommendations)
    )
    return rec_message