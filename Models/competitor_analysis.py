import numpy as np
from scipy.stats import invgamma, t  # Used for inverse gamma sampling and t-distribution predictions

def bayesian_competitor_analysis():
    # ----------------------------------------
    # 1. Data Simulation (Synthetic Data)
    # ----------------------------------------
    np.random.seed(42)
    n = 100
    time = np.arange(n)

    # Define competitor launch and duration of immediate impact
    comp_launch = 60
    comp_duration = 3  # Immediate competitor impact lasts for 3 periods

    # Competitor indicator active only during the impact window
    comp_effect = ((time >= comp_launch) & (time < comp_launch + comp_duration)).astype(int)
    # Time offset during competitor impact: how long since competitor entry
    comp_time = np.where(comp_effect == 1, time - comp_launch, 0)

    # True parameters for simulation
    alpha_true = 10
    beta1_true = 0.5     # baseline sales trend
    beta2_true = -4.0    # immediate drop when competitor enters
    beta3_true = -0.7    # additional downward slope during competitor window
    sigma_true = 2.0

    # Generate synthetic sales data with competitor impact
    sales = (
        alpha_true
        + beta1_true * time
        + beta2_true * comp_effect
        + beta3_true * comp_time
        + np.random.normal(0, sigma_true, size=n)
    )

    # ----------------------------------------
    # 2. Setup the Bayesian Model (Using Conjugacy)
    # ----------------------------------------
    # Design matrix X with columns: intercept, time, comp_effect, comp_time
    X = np.column_stack((np.ones(n), time, comp_effect, comp_time))
    p = X.shape[1]

    # Priors
    tau = 10.0  # prior std for betas
    a0 = 1.0    # prior shape for sigma^2 ~ InvGamma
    b0 = 1.0    # prior scale for sigma^2

    # Posterior for beta | sigma^2
    XtX = X.T.dot(X)
    Vn = np.linalg.inv(XtX + np.eye(p) / (tau**2))
    beta_n = Vn.dot(X.T).dot(sales)

    # Posterior for sigma^2
    a_n = a0 + n / 2
    b_n = b0 + 0.5 * (sales.dot(sales) - beta_n.dot((XtX + np.eye(p) / (tau**2)).dot(beta_n)))

    # ----------------------------------------
    # 3. Sampling from the Joint Posterior
    # ----------------------------------------
    num_samples = 5000
    beta_samples = np.zeros((num_samples, p))
    sigma2_samples = np.zeros(num_samples)

    for i in range(num_samples):
        sigma2 = invgamma.rvs(a=a_n, scale=b_n)
        sigma2_samples[i] = sigma2
        beta_samples[i, :] = np.random.multivariate_normal(beta_n, sigma2 * Vn)

    # ----------------------------------------
    # 4. Posterior Predictive Check
    # ----------------------------------------
    pred_mean = np.zeros(n)
    pred_lower = np.zeros(n)
    pred_upper = np.zeros(n)
    df = 2 * a_n

    for i in range(n):
        x_star = X[i, :]
        pred_mean[i] = x_star.dot(beta_n)
        scale = np.sqrt(b_n / a_n * (1 + x_star.dot(Vn).dot(x_star)))
        t_crit = t.ppf(0.975, df)
        pred_lower[i] = pred_mean[i] - t_crit * scale
        pred_upper[i] = pred_mean[i] + t_crit * scale

    # ----------------------------------------
    # 5. Estimate Parameters and Effects
    # ----------------------------------------
    beta_est = beta_samples.mean(axis=0)
    sigma_est = np.sqrt(sigma2_samples.mean())

    # Extract competitor impact parameters
    beta2_samples = beta_samples[:, 2]
    beta3_samples = beta_samples[:, 3]
    beta2_lower, beta2_upper = np.percentile(beta2_samples, [2.5, 97.5])
    beta3_lower, beta3_upper = np.percentile(beta3_samples, [2.5, 97.5])
    p_beta2_neg = np.mean(beta2_samples < 0)
    p_beta3_neg = np.mean(beta3_samples < 0)

    # ----------------------------------------
    # 6. Interpretation Helper
    # ----------------------------------------
    def interpret_competitor_effect(name, mean, lower, upper, p_negative):
        if p_negative >= 0.99:
            evidence = "overwhelming evidence of a negative competitor impact"
        elif p_negative >= 0.95:
            evidence = "strong evidence of downward pressure due to the competitor"
        elif p_negative >= 0.80:
            evidence = "moderate indication of competitor influence"
        elif p_negative >= 0.50:
            evidence = "some indication of a negative competitor effect, but uncertain"
        else:
            evidence = "little support for a negative competitor effect"
        return (
            f"For {name}, your sales reduced by around {mean:.2f} (95% CI: [{lower:.2f}, {upper:.2f}]) with a {p_negative:.1%} probability of a negative impact, indicating {evidence}."
        )

    conclusion_beta2 = interpret_competitor_effect(
        "Immediate Competitor Impact", beta_est[2], beta2_lower, beta2_upper, p_beta2_neg
    )
    conclusion_beta3 = interpret_competitor_effect(
        "Sustained Competitor Pressure", beta_est[3], beta3_lower, beta3_upper, p_beta3_neg
    )

    # ----------------------------------------
    # 7. Narrative Summary
    # ----------------------------------------
    narrative = (
        "## Competitor Analysis\n\n"
        "This report examines how the entry of a new competitor affected sales dynamics. We analyze both the immediate sales drop upon competitor launch and the longer-term sales trajectory during the competitive window.\n\n"
        "### Immediate Competitor Impact\n"
        f"{conclusion_beta2}\n\n"
        "### Sustained Competitor Pressure\n"
        f"{conclusion_beta3}\n\n"

    )
    return {"narrative": narrative}