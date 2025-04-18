import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import invgamma, t  # Used for inverse gamma sampling and t-distribution predictions

def bayesian_impact():
    # -------------------------------
    # 1. Data Simulation (Synthetic Data)
    # -------------------------------
    np.random.seed(42)
    n = 100
    time = np.arange(n)
    
    # Define promotion start and duration
    promo_start = 60
    promo_duration = 3  # Promotion is effective for 3 periods
    
    # Create promotion indicator that is active only during the effective period
    promo_effect = ((time >= promo_start) & (time < promo_start + promo_duration)).astype(int)
    # Create a time variable for the promotion period: time offset relative to promo_start when promotion is active
    promo_time = np.where(promo_effect == 1, time - promo_start, 0)
    
    # True parameters for simulation
    alpha_true = 10
    beta1_true = 0.5    # baseline trend slope
    beta2_true = 5      # immediate jump at promotion start (only during promo period)
    beta3_true = 0.8    # additional slope effect during the promo period
    sigma_true = 2.0
    
    # Generate synthetic sales data with a promotion effect that lasts only for promo_duration
    # After promo_start+promo_duration, the effect is turned off (back to baseline)
    sales = (alpha_true 
             + beta1_true * time 
             + beta2_true * promo_effect 
             + beta3_true * promo_time  # only nonzero during the promotion period
             + np.random.normal(0, sigma_true, size=n))
    
    # Plot the simulated sales data
    #plt.figure(figsize=(10, 5))
    #plt.plot(time, sales, 'o', label='Observed Sales')
    #plt.axvline(promo_start, color='r', linestyle='--', label='Promotion Start')
    #plt.axvline(promo_start + promo_duration, color='g', linestyle='--', label='Promotion End')
    #plt.xlabel('Time')
    #plt.ylabel('Sales')
    #plt.legend()
    #plt.title('Simulated Sales Data with Limited-Time Promotion Effect')
    #plt.show()
    
    # -------------------------------
    # 2. Setup the Bayesian Model (Using Conjugacy)
    # -------------------------------
    # Build design matrix X with columns:
    #   - Intercept
    #   - Time (baseline trend)
    #   - promo_effect (binary: 1 during promo period, 0 otherwise)
    #   - promo_time (time offset, nonzero only during the promo period)
    X = np.column_stack((np.ones(n), time, promo_effect, promo_time))
    p = X.shape[1]
    
    # Hyperparameters for the priors
    tau = 10.0  # prior standard deviation for beta coefficients
    a0 = 1.0    # prior shape for sigma^2 ~ InvGamma(a0, b0)
    b0 = 1.0    # prior scale for sigma^2
    
    # Compute posterior parameters for beta | sigma^2
    XtX = X.T.dot(X)
    Vn = np.linalg.inv(XtX + np.eye(p) / (tau**2))  # Posterior covariance factor (scaled by sigma^2)
    beta_n = Vn.dot(X.T).dot(sales)                   # Posterior mean for beta
    
    # Compute posterior parameters for sigma^2
    a_n = a0 + n / 2
    b_n = b0 + 0.5 * (sales.dot(sales) - beta_n.dot((XtX + np.eye(p) / (tau**2)).dot(beta_n)))
    
    #print("Posterior parameters:")
    #print("beta_n (posterior mean for beta):", beta_n)
    #print("Posterior covariance factor Vn (scaled by sigma^2):\n", Vn)
    #print("a_n (shape for sigma^2):", a_n)
    #print("b_n (scale for sigma^2):", b_n)
    
    # -------------------------------
    # 3. Sampling from the Joint Posterior Distribution
    # -------------------------------
    num_samples = 5000
    beta_samples = np.zeros((num_samples, p))
    sigma2_samples = np.zeros(num_samples)
    
    for i in range(num_samples):
        sigma2 = invgamma.rvs(a=a_n, scale=b_n)  # Sample sigma^2 from inverse gamma posterior
        sigma2_samples[i] = sigma2
        beta_samples[i, :] = np.random.multivariate_normal(beta_n, sigma2 * Vn)
    
    # -------------------------------
    # 4. Posterior Predictive Check
    # -------------------------------
    pred_mean = np.zeros(n)
    pred_lower = np.zeros(n)
    pred_upper = np.zeros(n)
    
    df = 2 * a_n  # Degrees of freedom for the t-distribution
    
    for i in range(n):
        x_star = X[i, :]
        pred_mean[i] = x_star.dot(beta_n)
        scale = np.sqrt(b_n / a_n * (1 + x_star.dot(Vn).dot(x_star)))
        t_crit = t.ppf(0.975, df)
        pred_lower[i] = pred_mean[i] - t_crit * scale
        pred_upper[i] = pred_mean[i] + t_crit * scale
    
    #plt.figure(figsize=(12, 6))
    #plt.plot(time, sales, 'o', label='Observed Sales')
    #plt.plot(time, pred_mean, 'r-', lw=2, label='Predictive Mean')
    #plt.fill_between(time, pred_lower, pred_upper, color='r', alpha=0.3, label='95% Credible Interval')
    #plt.axvline(promo_start, color='k', linestyle='--', label='Promotion Start')
    #plt.axvline(promo_start + promo_duration, color='g', linestyle='--', label='Promotion End')
    #plt.xlabel('Time')
    #plt.ylabel('Sales')
    #plt.title('Posterior Predictive Check')
    #plt.legend()
    #plt.show()
    
    # -------------------------------
    # 5. Diagnostics: Comparing True vs. Estimated Parameters
    # -------------------------------
    #print("\nTrue parameters:")
    #print("alpha =", alpha_true)
    #print("beta1 =", beta1_true)
    #print("beta2 =", beta2_true)
    #print("beta3 =", beta3_true)
    #print("sigma =", sigma_true)
    
    beta_est = beta_samples.mean(axis=0)
    sigma_est = np.sqrt(sigma2_samples.mean())
    
    #print("\nEstimated parameters (posterior means):")
    #print("alpha_est =", beta_est[0])
    #print("beta1_est =", beta_est[1])
    #print("beta2_est =", beta_est[2])
    #print("beta3_est =", beta_est[3])
    #print("sigma_est =", sigma_est)
    
    # -------------------------------
    # 6. Assessing the Impact of the Promotion Dynamically
    # -------------------------------
    # Extract samples for the effects of interest:
    #   - beta2 corresponds to the immediate jump (active only during the promo period)
    #   - beta3 corresponds to the additional slope during the promo period
    beta2_samples = beta_samples[:, 2]
    beta3_samples = beta_samples[:, 3]
    
    beta2_lower, beta2_upper = np.percentile(beta2_samples, [2.5, 97.5])
    beta3_lower, beta3_upper = np.percentile(beta3_samples, [2.5, 97.5])
    
    p_beta2_positive = np.mean(beta2_samples > 0)
    p_beta3_positive = np.mean(beta3_samples > 0)
    
    #print("\nPromotion Impact Analysis:")
    #print("Immediate jump (beta2):")
    #print("  Posterior mean =", beta_est[2])
    #print("  95% Credible Interval = [{:.2f}, {:.2f}]".format(beta2_lower, beta2_upper))
    #print("  P(beta2 > 0) = {:.1%}".format(p_beta2_positive))
    #print("\nAdditional slope change (beta3):")
    #print("  Posterior mean =", beta_est[3])
    #print("  95% Credible Interval = [{:.2f}, {:.2f}]".format(beta3_lower, beta3_upper))
    #print("  P(beta3 > 0) = {:.1%}".format(p_beta3_positive))
    
    # -------------------------------
    # 7. Dynamically Generating the Final Conclusion
    # -------------------------------
    def interpret_effect(effect_name, posterior_mean, lower, upper, p_positive):
        """
        Generate a dynamic, business-friendly conclusion based on the estimated effect,
        its 95% credible interval, and the posterior probability of a positive effect.
        """
        if p_positive >= 0.99:
            evidence = "overwhelming evidence that this effect is real"
        elif p_positive >= 0.95:
            evidence = "strong evidence supporting this impact"
        elif p_positive >= 0.80:
            evidence = "moderate evidence suggesting a positive influence"
        elif p_positive >= 0.50:
            evidence = "some indication of a positive effect, though it is less certain"
        else:
            evidence = "little to no support for a positive effect"
        
        # Craft a business-friendly conclusion statement
        conclusion = (f"For {effect_name}, our analysis shows an estimated effect of {posterior_mean:.2f} with a 95% "
                      f"credible interval ranging from {lower:.2f} to {upper:.2f}. There is a {p_positive:.1%} chance that "
                      f"this effect is positive, indicating {evidence}. This insight can help inform strategy and decision-making, "
                      f"ensuring that resources are focused on initiatives that yield the best short-term and long-term results.")
        return conclusion
    
    conclusion_beta2 = interpret_effect("Immediate Boost", beta_est[2], beta2_lower, beta2_upper, p_beta2_positive)
    conclusion_beta3 = interpret_effect("Sustained Growth", beta_est[3], beta3_lower, beta3_upper, p_beta3_positive)
    
    # -------------------------------
    # Create Narrative Impact Summary Object
    # -------------------------------
    narrative_impact_summary = (
        "## Impact Analysis\n\n"
        "This report provides a detailed analysis of how a recent limited-time promotion influenced sales. The evaluation focuses on two key areas: the immediate boost in sales at the start of the promotion and the longer-term sales trajectory following the promotion period.\n\n"
        "### Immediate Boost\n"
        f"{conclusion_beta2}\n\n"
        "### Sustained Growth\n"
        f"{conclusion_beta3}\n\n"
        "### Business Insights\n"
        "- **Rapid Revenue Opportunity:** The analysis indicates that the promotion triggered a significant short-term sales spike. This immediate boost can be leveraged to drive quick revenue growth.\n"
        "- **Temporary Impact:** The longer-term effect, while positive, is less pronounced. This suggests that while the promotion captured initial customer interest, follow-up strategies might be necessary to sustain growth.\n"
        "- **Strategic Recommendations:** Consider implementing complementary initiatives, such as loyalty programs or follow-up promotions, to extend the momentum generated by the initial boost.\n\n"
        "This comprehensive analysis is designed to support strategic decision-making and ensure that marketing investments are aligned with business objectives."
    )

    graph_data = {
        "time": time.tolist(),      # Time periods
        "sales": sales.tolist(),    # Simulated sales data
        "promo_start": promo_start, # Start of promotion
        "promo_end": promo_start + promo_duration  # End of promotion
    }

    result ={
        "narrative": narrative_impact_summary,
        "impact_graph_data":graph_data
    }
    
    # Return the narrative impact summary as the output of the function
    return result
