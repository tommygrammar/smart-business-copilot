import numpy as np
import pandas as pd
from Data.business_data import historical_data
from scipy.optimize import curve_fit
from scipy.stats import norm, beta
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import VAR


# =============================================================================
#  PARAMETERS
# =============================================================================
FORECAST_HORIZON = 365      # periods ahead to forecast
N_MONTE_CARLO    = 10000    # Monte Carlo paths
RISK_ALPHA       = 0.05     # 5% tail for VaR/ES
SEASONAL_PERIOD  = 12       # e.g. 12 for monthly data

# =============================================================================
#  LOAD & PREPARE DATA
# =============================================================================
df = pd.DataFrame({
    'Sales':             historical_data['sales'],
    'Revenue':           historical_data['revenue'],
    'Cost_Efficiency':   historical_data['cost_efficiency']
}, dtype=float)

n = len(df)
t = np.arange(n)

# =============================================================================
#  1. DECOMPOSE TREND & SEASONALITY
# =============================================================================
stl_results = {}
for col in df.columns:
    stl = STL(df[col], period=SEASONAL_PERIOD, robust=True).fit()
    stl_results[col] = stl
    df[f'{col}_detrended'] = df[col] - stl.seasonal

# =============================================================================
#  GROWTH MODEL FUNCTIONS
# =============================================================================
def exp_model(t, Q0, r):              return Q0 * np.exp(r * t)
def logistic_model(t, K, Q0, r):      return K / (1 + ((K-Q0)/Q0)*np.exp(-r*t))
def powerlaw_model(t, Q0, c, alpha):  return Q0 + c * t**alpha
def gompertz_model(t, A, B, C):       return A * np.exp(-B * np.exp(-C * t))
def richards_model(t, K, Q0, r, v):
    return K / (1 + ((K/Q0)**v - 1)*np.exp(-r*v*t))**(1/v)

def simulate_gbm(S0, mu, sigma, H, N):
    dt = 1.0
    eps = np.random.randn(N, H)
    increments = (mu - 0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*eps
    log_paths = np.cumsum(np.hstack([np.zeros((N,1)), increments]), axis=1)
    return S0 * np.exp(log_paths)

# =============================================================================
#  2. GLOBAL MODEL COMPARISON (BIC weights)
# =============================================================================
all_weights = {}
best_models = {}

for metric in ['Sales','Revenue','Cost_Efficiency']:
    y = df[f'{metric}_detrended'].values
    bic_scores = {}

    def fit_and_bic(fn, p0, bounds, k, name):
        try:
            popt, _ = curve_fit(fn, t, y, p0=p0, bounds=bounds, maxfev=20_000)
            resid = y - fn(t, *popt)
            sigma = np.sqrt(np.mean(resid**2))
            ll    = norm.logpdf(y, loc=fn(t,*popt), scale=sigma).sum()
            bic   = k * np.log(n) - 2*ll
            bic_scores[name] = bic
        except:
            bic_scores[name] = np.inf

    fit_and_bic(exp_model,      [y[0], np.log(y[-1]/y[0])/(n-1)],
                ([-np.inf,-np.inf],[np.inf,np.inf]), 2, 'Exponential')
    fit_and_bic(logistic_model, [y.max()*1.1, y[0], 0.1],
                ([0,0,0],[y.max()*10,y.max(),10]),      3, 'Logistic')
    fit_and_bic(powerlaw_model, [y[0], (y[-1]-y[0])/(n-1),1.0],
                ([0,0,0],[np.inf,np.inf,10]),            3, 'Power-Law')
    fit_and_bic(gompertz_model, [y.max(),1.0,0.1],
                ([0,0,0],[np.inf,np.inf,np.inf]),        3, 'Gompertz')
    fit_and_bic(richards_model, [y.max(),y[0],0.1,1.0],
                ([0,0,0,0],[np.inf,np.inf,np.inf,10]),   4, 'Richards')

    bics    = np.array(list(bic_scores.values()))
    delta   = bics - np.nanmin(bics)
    weights = np.exp(-0.5*delta) / np.nansum(np.exp(-0.5*delta))
    names   = list(bic_scores.keys())
    w_map   = dict(zip(names, weights))

    all_weights[metric] = w_map
    best_models[metric] = names[int(np.nanargmin(bics))]

# =============================================================================
#  3. CROSS-METRIC DYNAMICS (VAR)
# =============================================================================
var_data = df[[f'{m}_detrended' for m in ['Sales','Revenue','Cost_Efficiency']]]
var_mod  = VAR(var_data)
lag_sel  = var_mod.select_order(12).aic
var_res  = var_mod.fit(lag_sel)

# =============================================================================
#  4. RISK & DOWNTURN ANALYSIS (EWMA Vol)
# =============================================================================
risk = {}
for metric in ['Sales','Revenue','Cost_Efficiency']:
    s      = df[f'{metric}_detrended'].values
    ret    = np.log(s[1:]/s[:-1])
    mu, sigma = ret.mean(), ret.std(ddof=0)
    vol_ewma   = pd.Series(ret).ewm(span=20, adjust=False).std().iloc[-1]

    cum    = s / s[0]
    peak   = np.maximum.accumulate(cum)
    dd     = (cum - peak) / peak
    max_dd = dd.min()
    trough = dd.argmin()
    recov  = next((i for i in range(trough+1,len(cum)) if cum[i]>peak[trough]), None)

    var_es = {}
    for h in [1,5,10,FORECAST_HORIZON]:
        sim  = simulate_gbm(s[-1], mu, sigma, h, N_MONTE_CARLO)
        r_h  = (sim[:,-1] - s[-1]) / s[-1]
        v    = -np.quantile(r_h, RISK_ALPHA)
        e    = -r_h[r_h <= np.quantile(r_h, RISK_ALPHA)].mean()
        var_es[h] = {'VaR': v, 'ES': e}

    risk[metric] = {
        'mu': mu, 'sigma': sigma, 'vol_ewma': vol_ewma,
        'max_dd': max_dd, 'recovery': recov, 'var_es': var_es
    }


# =============================================================================
#  5. OUTPUT — DYNAMIC BUSINESS NARRATIVE
# =============================================================================

def generate_trend():
    # 5.1 Trend & Seasonality
    summary = (
        "## TREND & SEASONALITY\n\n"
    )
    for metric in ['Sales','Revenue','Cost_Efficiency']:
        stl        = stl_results[metric]
        start, end = stl.trend.iloc[0], stl.trend.iloc[-1]
        season_amp = stl.seasonal.std()
        pct_chg    = (end - start) / abs(start) * 100
        rel_season = season_amp / abs(end) * 100

        summary += (
            f"• **{metric}**: trend {start:.1f} → {end:.1f} ({pct_chg:.1f}% change); "
            f"seasonal amplitude ±{season_amp:.1f} ({rel_season:.1f}% of level)\n\n"
            f" **What this means:** the core {metric.lower()} baseline has "
            f"{'grown' if pct_chg>=0 else 'declined'} by {abs(pct_chg):.1f}% over the period, "
            f"while seasonal swings remain only ±{rel_season:.1f}% of that level.\n\n"
            f" **Business takeaway:** focus planning on the underlying trend; seasonality is minor.\n\n"
        )
    return summary

def generate_growth():

    # 5.2 Growth Dynamics
    summary = ( "## GROWTH DYNAMICS\n\n")
    for metric, wmap in all_weights.items():
        best   = best_models[metric]
        weight = wmap[best] * 100
        summary += f"• **{metric}**: best model = **{best}** ({weight:.1f}% weight)\n\n"
        #for name, w in wmap.items():
        #    summary += f"  {name:10s}: {w*100:5.1f}%\n\n"
        summary += "\n"
        if best == 'Exponential':
            summary += (
                " **Interpretation:** consistent percentage growth at a steady rate.\n\n"
                " **Takeaway:** maintain proportional investment to sustain momentum.\n\n"
            )
        elif best == 'Logistic':
            summary += (
                " **Interpretation:** S-curve growth nearing saturation.\n\n"
                " **Takeaway:** explore new markets or products to extend growth.\n\n"
            )
        elif best == 'Power-Law':
            summary += (
                " **Interpretation:** growth with decelerating pace.\n\n"
                " **Takeaway:** front-load efforts early; later stages require targeted pushes.\n\n"
            )
        else:  # Gompertz or Richards
            summary += (
                " **Interpretation:** rapid mid-phase acceleration then plateau.\n\n"
                " **Takeaway:** optimize efficiency and margins for mature growth phase.\n\n"
            )
    return summary

# 5.3 Interactions & Leads/Lags
def generate_interactions():
    summary = (f"## INTERACTIONS & LEADS/LAGS (VAR, {lag_sel} lags)\n\n")
    var_coefs = var_res.coefs[0]
    labels    = ['Sales','Revenue','Cost_Efficiency']
    summary += "### Dynamic cross-effect summary:\n\n"
    for i, metric in enumerate(labels):
        cross = [(j, var_coefs[i, j]) for j in range(len(labels)) if j != i]
        j, coef = max(cross, key=lambda x: abs(x[1]))
        other    = labels[j]
        direction= "increase" if coef > 0 else "decrease"
        summary += (
            f"- **{metric}**: strongest cross-driver is **{other}** "
            f"(lag-1 coef = {coef:.3f}), so a 1-unit rise in {other} one period ago "
            f"tends to {direction} {metric.lower()} by {abs(coef):.3f} today.\n"
        )
    summary += "\n"
    return summary

# 5.4 Risk & Downturn
def generate_risk():
    summary = ("## RISK & DOWNTURN\n\n")
    for metric, info in risk.items():
        mu, sigma, vol = info['mu'], info['sigma'], info['vol_ewma']
        max_dd, recov  = info['max_dd'], info['recovery']
        summary += (
            f"### {metric} :\n\n"
            f"- Avg return {mu*100:.2f}%, σ {sigma*100:.2f}%, "
            f"EWMA vol {vol*100:.2f}%\n\n"
        )
        summary += f" - Max drawdown: {max_dd*100:.1f}%\n\n"
        summary += (
            f" - Recovery: {recov} periods\n\n"
            if recov else " - Recovery: (no full recovery yet)\n\n"
        )
        summary += " - VaR/ES at 5%:\n"
        for h, ve in info['var_es'].items():
            summary += (
                f" • {h:3d}-period: VaR = {ve['VaR']*100:.2f}%, "
                f"ES = {ve['ES']*100:.2f}%\n\n"
            )
        summary += (
            f"\n **Interpretation:** the worst-case 5% drop is "
            f"{info['var_es'][1]['VaR']*100:.1f}% next period and "
            f"{info['var_es'][FORECAST_HORIZON]['VaR']*100:.1f}% over "
            f"{FORECAST_HORIZON} periods.\n\n"
            " **Takeaway:** ensure liquidity buffers cover these stress levels.\n\n"
        )
    return summary

# 5.5 Forecast Outlook
def generate_forecast_outlook():
    summary = (f"## {FORECAST_HORIZON}-PERIOD FORECAST\n\n")
    for metric in ['Sales','Revenue','Cost_Efficiency']:
        s0    = df[f'{metric}_detrended'].iloc[-1]
        mu, sigma = risk[metric]['mu'], risk[metric]['sigma']
        sim   = simulate_gbm(s0, mu, sigma, FORECAST_HORIZON, N_MONTE_CARLO)
        p10, p50, p90 = np.percentile(sim, [10,50,90], axis=0)
        delta_med = (p50[-1] - p50[0]) / abs(p50[0]) * 100
        delta_low = (p10[-1] - p10[0]) / abs(p10[0]) * 100
        delta_hi  = (p90[-1] - p90[0]) / abs(p90[0]) * 100

        summary += (
            f"• **{metric}**:\n"
            f"  Median: {p50[0]:.1f} → {p50[-1]:.1f} ({delta_med:+.1f}%)\n\n"
            f"  Conservative (10th): {p10[0]:.1f} → {p10[-1]:.1f} ({delta_low:+.1f}%)\n\n"
            f"  Optimistic (90th):  {p90[0]:.1f} → {p90[-1]:.1f} ({delta_hi:+.1f}%)\n\n"
            f"\n **Interpretation:** median change of {delta_med:+.1f}% over "
            f"{FORECAST_HORIZON} periods, with downside of {delta_low:+.1f}% "
            f"and upside of {delta_hi:+.1f}%.\n\n"
            " **Takeaway:** base plans on the median path but prepare for the 10th percentile.\n\n"
        )
    return summary

