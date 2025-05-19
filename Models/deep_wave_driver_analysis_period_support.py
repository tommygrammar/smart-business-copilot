import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import find_peaks, peak_widths
from Data.business_data import historical_data

data = historical_data

# ---------------------------
# Single Wave Analysis Function
# ---------------------------
def wave_analysis(series, num_predictions=10, beta=1e-4, num_iterations=1, fraction=0.01):
    """
    Analyzes a time series (wave) and returns:
      - Future predictions (via linear extrapolation)
      - Dominant frequency details (period, amplitude, interpretation)
      - The final refined (dominant) wave (after self-refinement)
    """
    # 1. Extract Root Wave using FFT
    fft_coeffs = fft(series)
    amplitudes = np.abs(fft_coeffs)
    dominant_idx = np.argmax(amplitudes[1:]) + 1  # Exclude DC component
    root_fft = np.zeros_like(fft_coeffs)
    root_fft[dominant_idx] = fft_coeffs[dominant_idx]
    if dominant_idx != 0:
        root_fft[-dominant_idx] = fft_coeffs[-dominant_idx]
    root_wave = np.real(ifft(root_fft))
    
    # Dominant frequency details
    freq = dominant_idx / len(series)  # cycles per day
    period = 1 / freq if freq != 0 else np.inf
    amplitude = amplitudes[dominant_idx]
    interpretation = f"Dominant frequency with period ~{period:.2f} days and amplitude ~{amplitude:.2f}"
    
    # 2. Create Variation Wave (average over 10 noise iterations)
    variation_dominant_fft = []
    noise_std_values = np.linspace(0.5, 2.0, 10)
    for noise_std in noise_std_values:
        noise = np.random.normal(0, noise_std, size=series.shape)
        noisy_data = series + noise
        fft_noisy = fft(noisy_data)
        amplitudes_noisy = np.abs(fft_noisy)
        dom_idx_noisy = np.argmax(amplitudes_noisy[1:]) + 1
        filtered_fft_noisy = np.zeros_like(fft_noisy)
        filtered_fft_noisy[dom_idx_noisy] = fft_noisy[dom_idx_noisy]
        if dom_idx_noisy != 0:
            filtered_fft_noisy[-dom_idx_noisy] = fft_noisy[-dom_idx_noisy]
        variation_dominant_fft.append(filtered_fft_noisy)
    variation_dominant_fft_avg = np.mean(variation_dominant_fft, axis=0)
    variation_wave = np.real(ifft(variation_dominant_fft_avg))
    
    # 3. Modify Root Wave using Variation Wave
    weight_root = 0.7
    weight_variation = 0.3
    modified_wave = weight_root * root_wave + weight_variation * variation_wave
    
    # 4. Multi-Wave Interference Processing (with preset cutoffs)
    cutoffs = [10, 50, 200]
    interference_components = []
    fft_mod = fft(modified_wave)
    fft_root = fft(root_wave)
    for cutoff in cutoffs:
        filtered_fft_mod = np.zeros_like(fft_mod)
        filtered_fft_mod[:cutoff] = fft_mod[:cutoff]
        filtered_fft_mod[-cutoff:] = fft_mod[-cutoff:]
        filtered_mod = np.real(ifft(filtered_fft_mod))
        
        filtered_fft_root = np.zeros_like(fft_root)
        filtered_fft_root[:cutoff] = fft_root[:cutoff]
        filtered_fft_root[-cutoff:] = fft_root[-cutoff:]
        filtered_root = np.real(ifft(filtered_fft_root))
        
        interference_component = 0.5 * filtered_mod + 0.5 * filtered_root
        interference_components.append(interference_component)
    weights_interference = [0.5, 0.3, 0.2]
    interference_value = sum(w * comp[-1] for w, comp in zip(weights_interference, interference_components))
    
    final_weight_modified = 0.7
    final_weight_interference = 0.3
    final_wave = final_weight_modified * modified_wave + final_weight_interference * np.full_like(modified_wave, interference_value)
    
    # 5. Self-Refinement Calibration
    refined_wave = final_wave.copy()
    n_points = len(series)
    for iteration in range(num_iterations):
        total_error = 0.0
        updated_wave = refined_wave.copy()
        for i in range(1, n_points):
            predicted_i = series[i-1] + (refined_wave[i] - refined_wave[i-1])
            target_i = series[i]
            error_i = predicted_i - target_i
            total_error += error_i**2
            updated_wave[i] = refined_wave[i] - beta * error_i
            updated_wave[i-1] = refined_wave[i-1] + beta * error_i
        mse = total_error / (n_points - 1)
        refined_wave = updated_wave
        if mse < 1e-4:
            break
    
    # 6. Predict Future Values using Linear Extrapolation
    predicted = []
    last_val = series[-1]
    d = refined_wave[-1] - refined_wave[-2]
    for k in range(5):
        predicted.append(last_val + k * d)
    predicted = np.array(predicted)
    
    # 7. Extract dominant refined wave: keep only a fraction of the top FFT coefficients
    fft_refined = fft(refined_wave)
    amplitudes_refined = np.abs(fft_refined)
    num_coeffs = len(fft_refined)
    num_to_keep = max(1, int(fraction * num_coeffs))
    sorted_indices = np.argsort(amplitudes_refined)[::-1]
    keep_indices = set(sorted_indices[:num_to_keep])
    filtered_fft = np.zeros_like(fft_refined)
    for idx in keep_indices:
        filtered_fft[idx] = fft_refined[idx]
        sym_idx = (-idx) % num_coeffs
        filtered_fft[sym_idx] = fft_refined[sym_idx]
    dominant_refined_wave = np.real(ifft(filtered_fft))
    
    return {
        "predicted": predicted,
        "dominant_details": {
            "frequency": freq,
            "period": period,
            "amplitude": amplitude,
            "interpretation": interpretation
        },
        "root_wave": root_wave,
        "variation_wave": variation_wave,
        "modified_wave": modified_wave,
        "refined_wave": dominant_refined_wave
    }

# ---------------------------
# Helper Function: Describe a Cycle Event in Plain Language
# ---------------------------
def cycle_behavior_description(series, refined_wave, factor_name, peak_distance, peak_prominence):
    """
    Produces an easy-to-understand description of a recent cycle event.
    It identifies when the rise started (from a trough), the day of the peak, and when it fell to a low.
    Days are numbered relative to the period under analysis.
    """
    peaks, _ = find_peaks(refined_wave, distance=peak_distance, prominence=peak_prominence)
    troughs, _ = find_peaks(-refined_wave, distance=peak_distance, prominence=peak_prominence)
    description = ""
    if len(peaks) > 0 and len(troughs) > 0:
        last_peak = peaks[-1]
        # Find the last trough before the peak and the first trough after the peak
        troughs_before = troughs[troughs < last_peak]
        troughs_after = troughs[troughs > last_peak]
        if len(troughs_before) > 0 and len(troughs_after) > 0:
            start_trough = troughs_before[-1]
            end_trough = troughs_after[0]
            description = (
                f"For {factor_name}, a clear cycle began rising around day {start_trough + 1}, "
                f"peaked on day {last_peak + 1}, and declined to a low by day {end_trough + 1}."
            )
        elif len(troughs_before) > 0:
            start_trough = troughs_before[-1]
            description = (
                f"For {factor_name}, the cycle started rising around day {start_trough + 1} and peaked on day {last_peak + 1}."
            )
        elif len(troughs_after) > 0:
            end_trough = troughs_after[0]
            description = (
                f"For {factor_name}, the cycle peaked on day {last_peak + 1} and then fell to a low around day {end_trough + 1}."
            )
        else:
            description = f"A significant peak for {factor_name} was observed on day {last_peak + 1}."
    else:
        description = f"Insufficient data to determine a clear cycle for {factor_name}."
    return description

# ---------------------------
# Wave Comparative Analysis Function
# ---------------------------
def wave_comparative_analysis(
    refined_wave1, refined_wave2, 
    label1="Signal 1", label2="Signal 2", 
    tolerance=5, peak_distance=30, peak_prominence=5
):
    """
    Compares the dominant (refined) waves for two signals and returns:
      - A friendly summary of the number of significant cycles detected in each signal.
      - The percentage overlap of cycles.
      - A dynamic conclusion.
    """
    # Find upward oscillations (peaks)
    peaks1, _ = find_peaks(refined_wave1, distance=peak_distance, prominence=peak_prominence)
    peaks2, _ = find_peaks(refined_wave2, distance=peak_distance, prominence=peak_prominence)
    count_peaks1 = len(peaks1)
    count_peaks2 = len(peaks2)
    
    if count_peaks1 > count_peaks2:
        more_label = label1
        fewer_label = label2
        more_peaks = peaks1
        fewer_peaks = peaks2
    else:
        more_label = label2
        fewer_label = label1
        more_peaks = peaks2
        fewer_peaks = peaks1
    
    # Find downward oscillations (troughs)
    troughs1, _ = find_peaks(-refined_wave1, distance=peak_distance, prominence=peak_prominence)
    troughs2, _ = find_peaks(-refined_wave2, distance=peak_distance, prominence=peak_prominence)
    count_troughs1 = len(troughs1)
    count_troughs2 = len(troughs2)
    
    if count_peaks1 > count_peaks2:
        more_troughs = troughs1
        fewer_troughs = troughs2
    else:
        more_troughs = troughs2
        fewer_troughs = troughs1
    
    # Compare upward cycles
    peak_matches = []
    for p in more_peaks:
        diffs = np.abs(fewer_peaks - p)
        if len(diffs) > 0 and np.min(diffs) <= tolerance:
            peak_matches.append(p)
    
    # Compare downward cycles
    trough_matches = []
    for t in more_troughs:
        diffs = np.abs(fewer_troughs - t)
        if len(diffs) > 0 and np.min(diffs) <= tolerance:
            trough_matches.append(t)
    
    # Calculate matching percentages
    up_ratio = len(peak_matches) / len(more_peaks) if len(more_peaks) > 0 else 0
    down_ratio = len(trough_matches) / len(more_troughs) if len(more_troughs) > 0 else 0
    avg_ratio = (up_ratio + down_ratio) / 2
    
    # Build summaries
    summary = (
        f"{label1} shows {count_peaks1} major upward cycles and {count_troughs1} downward cycles, "
        f"while {label2} shows {count_peaks2} upward and {count_troughs2} downward cycles. "
        f"Overall, {more_label} has more dominant cycles."
    )
    match_summary = (
        f"Approximately {int(avg_ratio*100)}% of the cycles in {more_label} overlap with those in {fewer_label}."
    )
    
    if avg_ratio > 0.8:
        conclusion = f"High overlap (80%). {more_label} is heavily influenced by {fewer_label}."
    elif avg_ratio > 0.4:
        conclusion = f"Moderate overlap (~{int(avg_ratio*100)}%). {more_label} is partially driven by {fewer_label}."
    else:
        conclusion = f"Low overlap (<40%). {more_label} is not solely driven by {fewer_label}."
    
    return {
        "friendly_summary": summary,
        "match_summary": match_summary,
        "dynamic_conclusion": conclusion
    }

# ---------------------------
# Configured Business Narrative Generation Function with Detailed Behavior Analysis and Dynamic Insights
# ---------------------------
def trend_generate_business_narrative(factor1, factor2, period_length):
    """
    Runs the wave analysis and comparative analysis for two factors and generates a business
    narrative summarizing:
      - Future predictions (trend forecasts)
      - Dominant cycle characteristics (period, amplitude, cycle duration)
      - Cycle event timing (when a cycle began rising, peaked, and declined)
      - Extreme values (highest and lowest observed values with their days)
      - Comparative insights (overlap between cycles)
      - And newly, an observed trend based on the last 5 data points in the dominant cycle.
    
    This automation produces dynamic insights similar to:
    
    60-Day Analysis
    --------------
    Trend Predictions and Forecasts:
      - Sales is predicted to decline, “dropping by roughly 0.68 units per period.”
        (Interpretation: Sales are trending downward over each forecasted period.)
      - Revenue is forecasted to fall, “decreasing by about 0.83 units per period.”
        (Interpretation: Revenue is declining at a slightly steeper rate.)
    
    Dominant Cycle Characteristics:
      For Sales:
        - Dominant cycle has a period of ~60.00 days with an amplitude ~280.75.
          (Interpretation: One full cycle every 60 days with high fluctuation.)
        - Significant peaks last on average 20.0 days; troughs last ~18.8 days.
          (Interpretation: Once rising, sales remain high for ~20 days, then decline for ~19 days.)
      For Revenue:
        - Similar period and amplitude.
        - Peaks last ~18.4 days; troughs last ~20.4 days.
    
    Cycle Event Timing and Extreme Values:
      - Sales: Cycle started rising around day 15 and peaked on day 45.
        Highest value: 472.74 on day 54, lowest: 423.53 on day 2.
        Observation: Last 5 data points show a “consistent drop.”
      - Revenue: Cycle started rising around day 16 and peaked on day 46.
        Highest value: 597.18 on day 59, lowest: 545.69 on day 6.
        Observation: Last 5 data points indicate a “steady decline.”
    
    Comparative Insights:
      - High overlap (80%) between sales and revenue cycles.
        (Interpretation: Revenue is heavily influenced by sales trends.)
    
    Parameters:
      - factor1, factor2: Data keys (e.g., "sales", "revenue")
      - period_length: Number of days to analyze (e.g., 60)
    
    Returns:
      A string with a comprehensive, plain-language narrative.
    """
    print(f"Analyzing the past {period_length} days")
    # Hardcoded analysis parameters
    num_predictions = 10
    beta = 1e-4
    num_iterations = 1
    fraction = 0.01
    tolerance = 5
    peak_distance = 30
    peak_prominence = 5

    # Extract series from global data
    series1 = np.array(data[factor1])
    series2 = np.array(data[factor2])
    
    # Slice series if period_length is provided
    if period_length is not None:
        series1 = series1[-period_length:]
        series2 = series2[-period_length:]
    
    # Perform wave analysis for each series
    wave1 = wave_analysis(series1, num_predictions, beta, num_iterations, fraction)
    wave2 = wave_analysis(series2, num_predictions, beta, num_iterations, fraction)
    
    # Detailed behavior analysis based on dominant wave
    details1 = wave1["dominant_details"]
    details2 = wave2["dominant_details"]
    behavior_statement1 = (
        f"For {factor1}, the dominant wave has a period of approximately {details1['period']:.2f} days "
        f"with an amplitude around {details1['amplitude']:.2f}, indicating strong cyclical activity."
    )
    behavior_statement2 = (
        f"For {factor2}, the dominant cycle repeats roughly every {details2['period']:.2f} days "
        f"with a similar amplitude, suggesting regular highs and lows."
    )
    
    # Cycle Duration Analysis (using peak widths)
    peaks1, _ = find_peaks(wave1["refined_wave"], distance=peak_distance, prominence=peak_prominence)
    troughs1, _ = find_peaks(-wave1["refined_wave"], distance=peak_distance, prominence=peak_prominence)
    if len(peaks1) > 0:
        widths_peaks1, _, _, _ = peak_widths(wave1["refined_wave"], peaks1, rel_height=0.5)
        avg_peak_duration1 = np.mean(widths_peaks1)
        peak_duration_statement1 = (
            f"In addition, significant peaks for {factor1} last on average {avg_peak_duration1:.1f} days."
        )
    else:
        peak_duration_statement1 = f"No clear peaks were detected for {factor1}."
        
    if len(troughs1) > 0:
        widths_troughs1, _, _, _ = peak_widths(-wave1["refined_wave"], troughs1, rel_height=0.5)
        avg_trough_duration1 = np.mean(widths_troughs1)
        trough_duration_statement1 = (
            f"Likewise, its lows (troughs) last about {avg_trough_duration1:.1f} days on average."
        )
    else:
        trough_duration_statement1 = f"No clear troughs were detected for {factor1}."
    
    peaks2, _ = find_peaks(wave2["refined_wave"], distance=peak_distance, prominence=peak_prominence)
    troughs2, _ = find_peaks(-wave2["refined_wave"], distance=peak_distance, prominence=peak_prominence)
    if len(peaks2) > 0:
        widths_peaks2, _, _, _ = peak_widths(wave2["refined_wave"], peaks2, rel_height=0.5)
        avg_peak_duration2 = np.mean(widths_peaks2)
        peak_duration_statement2 = (
            f"For {factor2}, peaks last on average {avg_peak_duration2:.1f} days."
        )
    else:
        peak_duration_statement2 = f"No clear peaks were detected for {factor2}."
        
    if len(troughs2) > 0:
        widths_troughs2, _, _, _ = peak_widths(-wave2["refined_wave"], troughs2, rel_height=0.5)
        avg_trough_duration2 = np.mean(widths_troughs2)
        trough_duration_statement2 = (
            f"The lows for {factor2} last about {avg_trough_duration2:.1f} days on average."
        )
    else:
        trough_duration_statement2 = f"No clear troughs were detected for {factor2}."
    
    detailed_cycle_statement = (
        f"{peak_duration_statement1} {trough_duration_statement1}\n\n"
        f"{peak_duration_statement2} {trough_duration_statement2}"
    )
    
    # Identify representative cycle event in plain language for each factor
    cycle_behavior_statement1 = cycle_behavior_description(series1, wave1["refined_wave"], factor1.capitalize(), peak_distance, peak_prominence)
    cycle_behavior_statement2 = cycle_behavior_description(series2, wave2["refined_wave"], factor2.capitalize(), peak_distance, peak_prominence)
    
    # Additional Analysis: Extreme Values
    max_val1 = np.max(series1)
    max_day1 = np.argmax(series1) + 1
    min_val1 = np.min(series1)
    min_day1 = np.argmin(series1) + 1
    additional_statement1 = (
        f"For {factor1}, the highest value observed was {max_val1:.2f} on day {max_day1}, "
        f"and the lowest was {min_val1:.2f} on day {min_day1}."
    )
    max_val2 = np.max(series2)
    max_day2 = np.argmax(series2) + 1
    min_val2 = np.min(series2)
    min_day2 = np.argmin(series2) + 1
    additional_statement2 = (
        f"For {factor2}, the highest value recorded was {max_val2:.2f} on day {max_day2}, "
        f"and the lowest was {min_val2:.2f} on day {min_day2}."
    )
    
    # Future trend predictions (from extrapolation)
    predicted1 = wave1["predicted"]
    predicted2 = wave2["predicted"]
    trend1 = predicted1[-1] - predicted1[-2]
    if trend1 > 0:
        forecast_statement1 = f"**{factor1.capitalize()}** is forecasted to continue its upward trend, increasing by about {trend1:.2f} units per period."
    elif trend1 < 0:
        forecast_statement1 = f"**{factor1.capitalize()}** is expected to decline, dropping by roughly {abs(trend1):.2f} units per period."
    else:
        forecast_statement1 = f"**{factor1.capitalize()}** appears stable in the upcoming periods."
    
    trend2 = predicted2[-1] - predicted2[-2]
    if trend2 > 0:
        forecast_statement2 = f"**{factor2.capitalize()}** is projected to rise, with an increase of approximately {trend2:.2f} units per period."
    elif trend2 < 0:
        forecast_statement2 = f"**{factor2.capitalize()}** is forecasted to fall, decreasing by about {abs(trend2):.2f} units per period."
    else:
        forecast_statement2 = f"**{factor2.capitalize()}** is likely to remain steady over the next few periods."
    
    # New Section: Observed Trend from the Last 5 Data Points in the Dominant Cycle
    dom_wave1 = wave1["refined_wave"]
    last5_dom1 = dom_wave1[-5:]
    diff1 = np.diff(last5_dom1)
    if np.all(diff1 > 0):
        observed_trend1 = f"Observation: The last 5 data points of the dominant cycle for {factor1} show a consistent rise."
    elif np.all(diff1 < 0):
        observed_trend1 = f"Observation: The last 5 data points of the dominant cycle for {factor1} show a consistent drop."
    else:
        observed_trend1 = f"Observation: The trend in the last 5 data points of the dominant cycle for {factor1} is mixed."
    
    dom_wave2 = wave2["refined_wave"]
    last5_dom2 = dom_wave2[-5:]
    diff2 = np.diff(last5_dom2)
    if np.all(diff2 > 0):
        observed_trend2 = f"Observation: The last 5 data points of the dominant cycle for {factor2} indicate a steady upward movement."
    elif np.all(diff2 < 0):
        observed_trend2 = f"Observation: The last 5 data points of the dominant cycle for {factor2} indicate a steady decline."
    else:
        observed_trend2 = f"Observation: The recent dominant cycle for {factor2} does not exhibit a clear trend."
    
    # Comparative Analysis
    comparison = wave_comparative_analysis(
        wave1["refined_wave"],
        wave2["refined_wave"],
        label1=factor1.capitalize(),
        label2=factor2.capitalize(),
        tolerance=tolerance,
        peak_distance=peak_distance,
        peak_prominence=peak_prominence
    )
    comp_statement = comparison["dynamic_conclusion"]
    
    # Compose the final narrative
    narrative = (
        "-----------------------------\n\n"
        f" # {factor1.capitalize()} and {factor2.capitalize()} for {period_length} days Summary:\n\n"
        f" ## Trend Predictions and Forecasts:\n"
        f"  - {forecast_statement1}\n"
        f"  - {forecast_statement2}\n\n"
        f" ## Dominant Cycle Characteristics:\n"
        f"  For {factor1}:\n"
        f"    - Dominant cycle period: {details1['period']:.2f} days\n"
        f"    - Amplitude: {details1['amplitude']:.2f}\n"
        f"    - {peak_duration_statement1} {trough_duration_statement1}\n\n"
        f"  For {factor2}:\n"
        f"    - Dominant cycle period: {details2['period']:.2f} days\n"
        f"    - Amplitude: {details2['amplitude']:.2f}\n"
        f"    - {peak_duration_statement2} {trough_duration_statement2}\n\n"
        f" ## Cycle Event Timing and Extreme Values:\n"
        f"  - {cycle_behavior_statement1}\n"
        f"  - {additional_statement1}\n"
        f"  - {observed_trend1}\n\n"
        f"  - {cycle_behavior_statement2}\n"
        f"  - {additional_statement2}\n"
        f"  - {observed_trend2}\n\n"
        f" ## Comparative Insights:\n"
        f"  - {comp_statement}\n"
        f"{comparison['friendly_summary']}\n"
        f"{comparison['match_summary']}\n"
    )
    
    return narrative


