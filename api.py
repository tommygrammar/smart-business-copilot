from flask import Flask, request, jsonify
from flask_cors import CORS
from Models.business_twin_sensitivity_model import business_twin
from Models.cash_flow import weekly_cashflow_sales_analysis
from Models.competitor_analysis import bayesian_competitor_analysis
from Models.deep_wave_driver_analysis_period_support import trend_generate_business_narrative
from Models.demand_analysis import demand_analysis
from Models.event_probability_models import calculate_event_probability
from Models.gradient_ascent_sensitivity_allocation_model import optimize
from Models.growth import generate_forecast_outlook, generate_growth, generate_risk, generate_interactions, generate_trend
from Models.impact_analysis_model import promotional_impact
from Models.official_business_summary import create_business_summary
from Models.probability_sell_strategy import sell
from Models.product_segmentation import type_shit
from Models.risk_model import run_risk_analysis
from Models.sales_forecast import forecast_sales
from Models.stockout import stockouts
from Models.strengths_weakness_assessment_model import find_strengths, find_weaknesses, losses
from Models.wave_driver_analysis import generate_business_narrative
from Models.sprob import spoint


app = Flask(__name__)
CORS(app)
#runs business twin simulations
@app.route('/twin', methods = ['POST'])
def query_endpoint():
    try:
        data = request.get_json()
        rate = float(data['twinrate'])
        factor = data['twinfactor'].replace(' ', '')        
        return business_twin(factor, rate ), 200
    except Exception as e:
        return jsonify(), 400

#runs cashflow probabilities
@app.route('/cashflowsales', methods = ['POST'])
def query_cashflow():
    try:
        data = request.get_json()
        unit_price = data['unit_price']
        initial_cash = data['initial_cash']
        forecast_days = data['forecast_days']
        
        return weekly_cashflow_sales_analysis(unit_price, initial_cash, forecast_days), 200
    except Exception as e:
        return jsonify(), 400

#runs competitor impact analysis
@app.route('/competitorimpact', methods = ['POST'])
def query_comptetitor():
    try:
        data = request.get_json()
        
        return bayesian_competitor_analysis(), 200
    except Exception as e:
        return jsonify(), 400

 #runs periodic deep wave analysis   
@app.route('/deepwave', methods = ['POST'])
def query_deepwave():
    try:
        data = request.get_json()
        factor1=data['factor1'].replace(' ', '')
        factor2=data['factor2'].replace(' ', '')
        period_length = data['period_length']
        
        return trend_generate_business_narrative(factor1, factor2, period_length), 200
    except Exception as e:
        return jsonify(), 400
    
#runs demand analysis
@app.route('/demand', methods = ['POST'])
def query_demand():
    try:
        data = request.get_json()
        days = data['n_days']
        product = data['product'].replace(' ', '')

        
        return demand_analysis(days, product), 200
    except Exception as e:
        return jsonify(), 400

 #runs event probabilities   
@app.route('/eventprobability', methods = ['POST'])
def query_eventprobability():
    try:
        data = request.get_json()
        factor = data['factor'].replace(' ', '')
        target = data['target']
        period = data['period']

        
        return (calculate_event_probability(factor, target, period, use_gbm=False, use_rolling_window=True, plot_residuals=False)), 200
    except Exception as e:
        return jsonify(), 400
#resource allocation optimizations
@app.route('/optimize', methods = ['POST'])
def query_optimize():
    try:
        data = request.get_json()
        mode = data['mode'].replace(' ', '')
        target_revenue = data['target_revenue']
        s_change = data['s_change']
        m_change = data['m_change']

        
        return optimize(mode, target_revenue,m_change,  s_change), 200
    except Exception as e:
        return jsonify(), 400  

#trend analytics
@app.route('/trend', methods = ['POST'])
def query_trend():

    try:
        
        data = request.get_json()
        factor1 = data['trendFactor1'].replace(' ', '')        
        return generate_trend(factor1), 200
    except Exception as e:
        return jsonify(), 400 

#growth analytics
@app.route('/growth', methods = ['POST'])
def query_growth():
    try:
        data = request.get_json()
        factor1 = data['growthFactor1'].replace(' ', '')
        return generate_growth(factor1), 200
    except Exception as e:
        return jsonify(), 400 
#kpi interactions
@app.route('/kpiinteractions', methods = ['POST'])
def query_interactions():
    try:
        data = request.get_json()
        factor1 = data['iFactor1'].replace(' ', '')
        factor2 = data['iFactor2'].replace(' ', '')
        
        
        return generate_interactions(factor1, factor2), 200
    except Exception as e:
        return jsonify(), 400 
#periodic kpi risks  
@app.route('/kpirisk', methods = ['POST'])
def query_kpirisk():
    try:
        data = request.get_json()
        forecast_horizon = data['rHorizon']
        factor1 = data['rFactor1'].replace(' ', '')        
        return (generate_risk(forecast_horizon,factor1))[0], 200
    except Exception as e:
        return jsonify(), 400 
#periodic kpi forecasts
@app.route('/kpiforecast', methods = ['POST'])
def query_kpiforecast():
    try:
        data = request.get_json()
        forecast_horizon = data['fHorizon']
        factor1 = data['fFactor1'].replace(' ', '')
        
        return generate_forecast_outlook(forecast_horizon,factor1), 200
    except Exception as e:
        return jsonify(), 400 

#promotional impact analysis
@app.route('/promotionalimpact', methods = ['POST'])
def promotionalimpact():
    try:
        data = request.get_json()
        
        return promotional_impact(), 200
    except Exception as e:
        return jsonify(), 400 

#business summary
@app.route('/summary', methods = ['POST'])
def summary():
    try:
        data = request.get_json()
        
        return create_business_summary(), 200
    except Exception as e:
        return jsonify(), 400 

#selling model
@app.route('/sellingpointers', methods = ['POST'])
def sellpointers():
    try:
        data = request.get_json()
        
        return sell(), 200
    except Exception as e:
        return jsonify(), 400 
    
#products segmentations
@app.route('/productsegmentation', methods = ['POST'])
def productsegmentation():
    try:
        data = request.get_json()
        
        return type_shit(), 200
    except Exception as e:
        return jsonify(), 400 
#
@app.route('/targetrisk', methods = ['POST'])
def targetrisk():
    try:
        data = request.get_json()
        factor = data['factor'].replace(' ', '')
        target = data['target']
        
        return run_risk_analysis(factor, target), 200
    except Exception as e:
        return jsonify(), 400 

@app.route('/salesforecast', methods = ['POST'])
def saleforecast():
    try:
        data = request.get_json()
        horizon = data['horizon']

        
        return forecast_sales(horizon), 200
    except Exception as e:
        return jsonify(), 400 

@app.route('/stockout', methods = ['POST'])
def stockout():
    try:
        data = request.get_json()
        days = data['days']
        
        return stockouts(days), 200
    except Exception as e:
        return jsonify(), 400 

@app.route('/assess', methods = ['POST'])
def assess():
    try:
        data = request.get_json()
        mode = data['mode'].replace(' ', '')
        factor = data['factor'].replace(' ', '')

        if mode == 'strength':
            return find_strengths(factor), 200
        if mode == 'weakness':
            return find_weaknesses(factor), 200
        if mode == 'loss':
            return losses(factor), 200

    except Exception as e:
        return jsonify(), 400 

@app.route('/driver', methods = ['POST'])
def driver():
    try:
        data = request.get_json()
        factor1 = data['factor1'].replace(' ', '')
        factor2 = data['factor2'].replace(' ', '')
        
        return generate_business_narrative(factor1, factor2), 200
    except Exception as e:
        return jsonify(), 400 
    
#runs business twin simulations
@app.route('/spointers', methods=['POST'])
def spointers():
    try:
        payload = request.get_json(silent=True) or {}
        result = spoint()        # <-- now returns a dict
        return jsonify(result), 200
    except Exception as e:
        return jsonify({ 'error': str(e) }), 400


app.run(debug = True, port=5000)
