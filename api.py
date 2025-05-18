from flask import Flask, request, jsonify
from flask_cors import CORS
from Models.business_twin_sensitivity_model import business_twin
from Models.cash_flow import weekly_cashflow_sales_analysis
from Models.competitor_analysis import bayesian_competitor_analysis
from Models.deep_wave_driver_analysis_period_support import trend_generate_business_narrative

from Models.event_probability_models import calculate_event_probability
from Models.gradient_ascent_sensitivity_allocation_model import optimize
from Models.growth import generate_forecast_outlook, generate_growth, generate_risk, generate_trend
from Models.impact_analysis_model import promotional_impact
from Models.official_business_summary import create_business_summary
from Models.probability_sell_strategy import sell

from Models.sales_forecast import forecast_sales
from Models.stockout import stockouts
from Models.strengths_weakness_assessment_model import find_strengths, find_weaknesses, losses
from Models.wave_driver_analysis import generate_business_narrative
from Models.sprob import spoint
from Models.AnomalyDetection import AnomalyDetector
import pandas as pd


app = Flask(__name__)
CORS(app)
#runs business twin simulations
@app.route('/twin', methods = ['POST'])
def query_endpoint():
    
    try:
        data = request.get_json()
        rate = float(data['twinrate'])
        factor = (data['twinfactor'].replace(' ', '')).lower()
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
        factor1=(data['factor1'].replace(' ', '')).lower()
        factor2=(data['factor2'].replace(' ', '')).lower()
        period_length = data['period_length']
        
        return trend_generate_business_narrative(factor1, factor2, period_length), 200
    except Exception as e:
        return jsonify(), 400
    
 #runs event probabilities   
@app.route('/eventprobability', methods = ['POST'])
def query_eventprobability():
    try:
        data = request.get_json()
        factor = (data['factor'].replace(' ', '')).lower()
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
        mode = (data['mode'].replace(' ', '')).lower()
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
        factor1 = (data['trendFactor1'].replace(' ', '')).lower()
        return generate_trend(factor1), 200
    except Exception as e:
        return jsonify(), 400 

#growth analytics
@app.route('/growth', methods = ['POST'])
def query_growth():
    try:
        data = request.get_json()
        factor1 = (data['growthFactor1'].replace(' ', '')).lower()
        return generate_growth(factor1), 200
    except Exception as e:
        return jsonify(), 400 

#periodic kpi risks  
@app.route('/kpirisk', methods = ['POST'])
def query_kpirisk():
    data = request.get_json() 
    print(data)    
    try:
               
              
        forecast_horizon = data['horizon']
        factor1 = (data['factor'].replace(' ', '')).lower()
        return (generate_risk(forecast_horizon,factor1))[0], 200
    except Exception as e:
        return e, 400 
#periodic kpi forecasts
@app.route('/kpiforecast', methods = ['POST'])
def query_kpiforecast():
    try:
        data = request.get_json()
        forecast_horizon = data['horizon']
        factor1 = (data['factor'].replace(' ', '')).lower()
        
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
        factor1 = (data['factor1'].replace(' ', '')).lower()
        factor2 = (data['factor2'].replace(' ', '')).lower()
        
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
        
@app.route('/anomalize', methods=['POST'])
def detect_anomaly():
    try:
        # Parse input form data
        date_column = (request.form.get('date_column')).capitalize()
        target_column = (request.form.get('target_column')).capitalize()
        days = int(request.form.get('days'))

        # Ensure file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'Missing Excel file in request'}), 400

        file = request.files['file']
        

        # Read Excel into DataFrame
        df = pd.read_excel(file, parse_dates=[date_column])

        # Sort by date and limit to the last `days`
        df.sort_values(by=date_column, inplace=True)
        df = df[-days:]  # take last `days` rows

        # Set the date column as index
        df.set_index(date_column, inplace=True)

        # Extract the series to analyze
        series = df[target_column]

        # Run anomaly detection
        detector = AnomalyDetector()
        resid = (detector.decompose(series.values))[2]
        z_scores = detector.rolling_mad_zscore(resid)
        daily_anoms = detector.flag_point_anomalies()
        weekly_scores = detector.aggregate_window_scores()
        weekly_anoms = detector.flag_window_anomalies()

        return jsonify({
            "dates": series.index.strftime('%Y-%m-%d').tolist(),
            "sales": series.tolist(),
            "residual": resid.tolist(),
            "z_scores": z_scores.tolist(),
            "daily_anomalies": daily_anoms.tolist(),
            "weekly_scores": weekly_scores.tolist(),
            "weekly_anomalies": weekly_anoms.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

#app.run(debug = True)