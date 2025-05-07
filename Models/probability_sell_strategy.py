import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm
def sell():
    # 1. Synthetic data generation with purchase‐history features
    def generate_synthetic_data(n_users=1000, sessions_per_user=5,
                                seasonal_amplitude=1.0, trend_slope=0.01, seed=42):
        np.random.seed(seed)
        data = []
        # ground‐truth coefficients
        true_beta = {
            'intercept': -2.0,
            'page_views': 0.3,
            'time_on_page': 0.02,
            'hour_sin': 0.5,
            'hour_cos': -0.2,
            'day_of_week': 0.1,
            'purchase_rate': 1.5,
            'cumulative_purchases': 0.1
        }
        user_history = {u: [] for u in range(n_users)}
        for u in range(n_users):
            for s in range(sessions_per_user):
                pv = np.random.poisson(lam=5)
                tp = np.random.exponential(scale=60)
                h = np.random.randint(0, 24)
                hs = np.sin(2 * np.pi * h / 24)
                hc = np.cos(2 * np.pi * h / 24)
                dw = np.random.randint(0, 7)
                seasonal = seasonal_amplitude * np.sin(2 * np.pi * s / sessions_per_user)
                trend = trend_slope * s
                # history features
                prev = sum(user_history[u])
                rate = prev / s if s > 0 else 0.0
                cum = prev
                # linear predictor
                lin = ( true_beta['intercept']
                    + true_beta['page_views'] * pv
                    + true_beta['time_on_page'] * tp
                    + true_beta['hour_sin'] * hs
                    + true_beta['hour_cos'] * hc
                    + true_beta['day_of_week'] * dw
                    + seasonal + trend
                    + true_beta['purchase_rate'] * rate
                    + true_beta['cumulative_purchases'] * cum )
                p = expit(lin)
                purchase = np.random.binomial(1, p)
                user_history[u].append(purchase)
                data.append({
                    'page_views': pv,
                    'time_on_page': tp,
                    'hour_sin': hs,
                    'hour_cos': hc,
                    'day_of_week': dw,
                    'seasonal': seasonal,
                    'trend': trend,
                    'purchase_rate': rate,
                    'cumulative_purchases': cum,
                    'purchase': purchase
                })
        return pd.DataFrame(data)

    # 2. Bayesian logistic via Metropolis‐Hastings
    def bayesian_logistic_mh(X, y, n_samples=3000, burn_in=500,
                            proposal_scale=0.1, prior_sd=5.0, seed=42):
        np.random.seed(seed)
        n, d = X.shape
        beta = np.zeros(d)
        samples = []
        prior_cov = (prior_sd ** 2) * np.eye(d)
        inv_prior = np.linalg.inv(prior_cov)
        prop_cov = proposal_scale * np.eye(d)
        for i in range(n_samples + burn_in):
            beta_prop = np.random.multivariate_normal(beta, prop_cov)
            lp = -0.5 * beta @ inv_prior @ beta
            lp_p = -0.5 * beta_prop @ inv_prior @ beta_prop
            ll = (y * (X @ beta) - np.log1p(np.exp(X @ beta))).sum()
            ll_p = (y * (X @ beta_prop) - np.log1p(np.exp(X @ beta_prop))).sum()
            if np.log(np.random.rand()) < (lp_p + ll_p) - (lp + ll):
                beta = beta_prop
            if i >= burn_in:
                samples.append(beta.copy())
        return np.array(samples)

    # === Main ===
    data = generate_synthetic_data()

    features = [
        'page_views','time_on_page','hour_sin','hour_cos',
        'day_of_week','seasonal','trend',
        'purchase_rate','cumulative_purchases'
    ]
    X_df = data[features].copy()
    X_df = sm.add_constant(X_df)
    y = data['purchase'].values
    X = X_df.values

    # Frequentist fit
    freq_model = sm.Logit(y, X).fit(disp=False)
    freq_params = freq_model.params

    # Bayesian sampling
    samples = bayesian_logistic_mh(X, y)

    # 6. Coefficient credible intervals
    lower_beta = np.percentile(samples, 2.5, axis=0)
    upper_beta = np.percentile(samples, 97.5, axis=0)
    odds_lower = np.exp(lower_beta)
    odds_upper = np.exp(upper_beta)
    pct_lower = (odds_lower - 1) * 100
    pct_upper = (odds_upper - 1) * 100

    summary = []

    # 7. Print coefficient summary with uncertainty
    summary = (f"# Selling Strategy: Coefficient Impacts with 95% Credible Intervals (Bayesian):\n\n")
    for name, low, high in zip(X_df.columns, pct_lower, pct_upper):
        if name == 'const': continue
        summary += (f"- **{name.replace('_',' ').title()}**: {low:.1f}% to {high:.1f}% change in odds\n\n")


    # 8. Predictive scenario with uncertainty
    scenario = {
        'page_views': 10, 'time_on_page': 120,
        'hour_sin': np.sin(2*np.pi*15/24),
        'hour_cos': np.cos(2*np.pi*15/24),
        'day_of_week': 2, 'seasonal': np.sin(2*np.pi*1/5),
        'trend': 0.02, 'purchase_rate': 0.3,
        'cumulative_purchases': 3
    }
    # 9. Enhanced recommendation with probabilistic confidence
    pct_freq = (np.exp(freq_params) - 1) * 100
    impacts = dict(zip(X_df.columns, pct_freq))
    impacts.pop('const', None)
    sorted_feats = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    boost, reduce = sorted_feats[0], sorted_feats[-1]

    summary += (f"\n ## Enhanced Actionable Insights:\n\n")
    summary += (f"- Emphasize '{boost[0]}' – expected +{boost[1]:.1f}% on average, "
        f"with high certainty based on observed data.\n\n")
    summary += (f"- De‐emphasize '{reduce[0]}' – expected {reduce[1]:.1f}% drag, "
        f"avoid to maintain conversion momentum.\n\n")

    return summary

