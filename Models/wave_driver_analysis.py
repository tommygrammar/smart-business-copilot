import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import find_peaks
from Data.business_data import historical_data

data = historical_data
# ---------------------------
# Single Wave Analysis Function
# ---------------------------
def wave_analysis(series, num_predictions=10, beta=1e-4, num_iterations=1, fraction=0.01):
    """
    Analyzes a time series (wave) and returns:
      - Future predictions (via linear extrapolation)
      - Dominant frequency details
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
# Configured Business Narrative Generation Function
# ---------------------------

def generate_business_narrative(factor1, factor2):
    """
    Runs the wave analysis and comparative analysis for two factors from a global
    data source and generates a business narrative summarizing the predictions and cyclic comparisons.
    In addition, collects the final dominant refined wave data for both factors under the key 'wave_graph_data'.
    
    Returns a dictionary with:
         - 'narrative': A string with the dynamic business narrative.
         - 'wave_graph_data': A dict with the final refined wave data for both factors.
    """
    # Hardcoded analysis parameters
    num_predictions = 10
    beta = 1e-4
    num_iterations = 1
    fraction = 0.01
    tolerance = 5
    peak_distance = 30
    peak_prominence = 5

    # Initialize default wave data (empty lists) in case of error.
    default_wave = {
        "refined_wave": [],
        "predicted": [0, 0]  # Ensure at least two points for trend calculation.
    }
    
    try:
        # Extract series from the global data dictionary
        series1 = np.array(data[factor1])
        series2 = np.array(data[factor2])
        
        # Perform wave analysis for each series
        wave1 = wave_analysis(series1, num_predictions, beta, num_iterations, fraction)
        wave2 = wave_analysis(series2, num_predictions, beta, num_iterations, fraction)
    except Exception as e:
        print("Error in wave analysis:", e)
        wave1 = default_wave
        wave2 = default_wave

    # Make sure we have the necessary keys and data.
    if "refined_wave" not in wave1 or not isinstance(wave1["refined_wave"], np.ndarray):
        wave1["refined_wave"] = np.array([])
    if "refined_wave" not in wave2 or not isinstance(wave2["refined_wave"], np.ndarray):
        wave2["refined_wave"] = np.array([])

    # Get future predictions and determine trends safely.
    predicted1 = wave1.get("predicted", [0, 0])
    predicted2 = wave2.get("predicted", [0, 0])
    if len(predicted1) < 2:
        predicted1 = [0, 0]
    if len(predicted2) < 2:
        predicted2 = [0, 0]
    
    trend1 = predicted1[-1] - predicted1[-2]
    if trend1 > 0:
        statement1 = f"{factor1.capitalize()} is forecasted to continue its upward trajectory, with an average increase of {trend1:.2f} units per period.\n\n"
    elif trend1 < 0:
        statement1 = f"{factor1.capitalize()} is forecasted to decline, with an average decrease of {abs(trend1):.2f} units per period.\n\n"
    else:
        statement1 = f"{factor1.capitalize()} is expected to remain steady in the upcoming periods.\n\n"
    
    trend2 = predicted2[-1] - predicted2[-2]
    if trend2 > 0:
        statement2 = f"{factor2.capitalize()} is projected to rise, showing an average increase of {trend2:.2f} units per period.\n\n"
    elif trend2 < 0:
        statement2 = f"{factor2.capitalize()} is projected to fall, with an average drop of {abs(trend2):.2f} units per period.\n\n"
    else:
        statement2 = f"{factor2.capitalize()} is anticipated to remain constant over the next few periods.\n\n"
    
    # Perform comparative analysis if possible.
    try:
        comparison = wave_comparative_analysis(
            wave1["refined_wave"],
            wave2["refined_wave"],
            label1=factor1.capitalize(),
            label2=factor2.capitalize(),
            tolerance=tolerance,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence
        )
        comp_statement = comparison.get("dynamic_conclusion", "no significant cyclic relationship was detected.")
    except Exception as e:
        print("Error in comparative analysis:", e)
        comp_statement = "insufficient data to determine a cyclic relationship."

    narrative = (
        "-----------------------------\n\n"
        f"### {factor1.capitalize()} and {factor2.capitalize()} Summary:\n\n"
        f"{statement1}{statement2}"
        f"Furthermore, the cyclic patterns between {factor1} and {factor2} indicate that {comp_statement.lower()}"
    )
    
    result = {
        "narrative": narrative,
        "wave_graph_data": {
            factor1: wave1["refined_wave"].tolist() if wave1["refined_wave"].size > 0 else [],
            factor2: wave2["refined_wave"].tolist() if wave2["refined_wave"].size > 0 else []
        }
    }
    
    return result
# ---------------------------
# Example Usage:
# ---------------------------
# Make sure 'data' is defined globally, for example:
# data = {
#     "sales": np.random.randn(100),
#     "revenue": np.random.randn(100)
# }
#
# Now, simply call:
#print(f"{generate_business_narrative("sales", "revenue")}\n\n")


