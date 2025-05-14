import numpy as np
import pandas as pd
from scipy.special import expit
from datetime import datetime, timedelta

def spoint():

    def generate_synthetic_data(
        n_users=500,
        n_days=730,
        avg_sessions_per_user_per_day=0.1,
        trend_slope=0.0005,
        holiday_prob=0.02,
        promo_prob=0.05,
        seed=42
    ):
        np.random.seed(seed)
        data = []
        true_beta = {
            'intercept': -3.0,
            'page_views': 0.20,
            'time_on_page': 0.015,
            'day_of_week': 0.05,
            'is_weekend': -0.20,
            'holiday': 0.50,
            'promotion': 0.40,
            'peak_hour': 0.30,
            'trend': 1.00,
            'purchase_rate': 1.50,
            'cumulative_purchases': 0.10
        }
        user_history_count = {u: 0 for u in range(n_users)}
        user_sessions     = {u: 0 for u in range(n_users)}
        start_date = datetime.today() - timedelta(days=n_days)

        for day_offset in range(n_days):
            current_date = start_date + timedelta(days=day_offset)
            dow        = current_date.weekday()                 # 0=Mon … 6=Sun
            is_weekend = 1 if dow >= 5 else 0
            holiday    = 1 if np.random.rand() < holiday_prob else 0
            promotion  = 1 if np.random.rand() < promo_prob   else 0
            trend      = trend_slope * day_offset

            for u in range(n_users):
                n_sess = np.random.poisson(lam=avg_sessions_per_user_per_day)
                for _ in range(n_sess):
                    user_sessions[u] += 1
                    prev_purchases = user_history_count[u]
                    purchase_rate  = prev_purchases / (user_sessions[u] - 1) if user_sessions[u] > 1 else 0.0
                    cum_purchases  = prev_purchases

                    pv   = np.random.poisson(lam=5)
                    tp   = np.random.exponential(scale=60)
                    hour = np.random.randint(0, 24)
                    peak_hour = 1 if 18 <= hour < 22 else 0

                    linear_part = (
                        true_beta['intercept']
                        + true_beta['page_views'] * pv
                        + true_beta['time_on_page'] * tp
                        + true_beta['day_of_week'] * dow
                        + true_beta['is_weekend'] * is_weekend
                        + true_beta['holiday'] * holiday
                        + true_beta['promotion'] * promotion
                        + true_beta['peak_hour'] * peak_hour
                        + true_beta['trend'] * trend
                        + true_beta['purchase_rate'] * purchase_rate
                        + true_beta['cumulative_purchases'] * cum_purchases
                    )

                    p = expit(linear_part)
                    purchase = np.random.binomial(1, p)
                    user_history_count[u] += purchase

                    data.append({
                        'date': current_date,
                        'user_id': u,
                        'page_views': pv,
                        'time_on_page': tp,
                        'day_of_week': dow,
                        'is_weekend': is_weekend,
                        'holiday': holiday,
                        'promotion': promotion,
                        'peak_hour': peak_hour,
                        'trend': trend,
                        'purchase_rate': purchase_rate,
                        'cumulative_purchases': cum_purchases,
                        'purchase': purchase
                    })

        return pd.DataFrame(data)

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

    # === Main analysis ===
    df = generate_synthetic_data()

    features = [
        'page_views',
        'time_on_page',
        'day_of_week',
        'is_weekend',
        'holiday',
        'promotion',
        'peak_hour',
        'trend',
        'purchase_rate',
        'cumulative_purchases'
    ]
    X_df = df[features].copy()
    X_df.insert(0, 'const', 1.0)

    y = df['purchase'].values
    X = X_df.values

    samples = bayesian_logistic_mh(X, y)
    lower_beta  = np.percentile(samples, 2.5, axis=0)
    median_beta = np.percentile(samples, 50, axis=0)
    upper_beta  = np.percentile(samples, 97.5, axis=0)

    odds_lower   = np.exp(lower_beta)
    odds_median  = np.exp(median_beta)
    odds_upper   = np.exp(upper_beta)

    pct_lower   = (odds_lower - 1) * 100
    pct_median  = (odds_median - 1) * 100
    pct_upper   = (odds_upper - 1) * 100

    # Identify top & bottom drivers
    impacts = {
        name: pct_median[i]
        for i, name in enumerate(X_df.columns) if name != 'const'
    }
    sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    top_factor, top_val       = sorted_impacts[0]
    bottom_factor, bottom_val = sorted_impacts[-1]

    # Build JSON‑serializable output
    factors = []
    for i, name in enumerate(X_df.columns):
        if name == 'const':
            continue
        low  = pct_lower[i]
        avg  = pct_median[i]
        high = pct_upper[i]
        label = name.replace('_', ' ').title()

        if name == top_factor:
            insight = f"Focusing on {label.lower()} delivers the biggest boost."
        elif name == bottom_factor:
            insight = f"{label} may slow things down—watch it closely."
        else:
            insight = "Changes here yield moderate gains—consider after the top drivers."

        factors.append({
            'name':        label,
            'typical_effect': float(avg),
            'range_min':   float(low),
            'range_max':   float(high),
            'insight':     insight
        })

    return { 'factors': factors }
