# --- anomaly_detection.py ---
"""
Module: anomaly_detection.py
Defines AnomalyDetector for point and window-level anomaly detection.
"""
import numpy as np

from statsmodels.tsa.seasonal import STL

class AnomalyDetector:
    def __init__(
        self,
        period: int = 7,
        stl_robust: bool = True,
        rolling_window: int = 30,
        point_threshold: float = 3.5,
        agg_window: int = 7,
        agg_percentile: float = 95.0,
        agg_func=np.max
    ):
        self.period = period
        self.stl_robust = stl_robust
        self.rolling_window = rolling_window
        self.point_threshold = point_threshold
        self.agg_window = agg_window
        self.agg_percentile = agg_percentile
        self.agg_func = agg_func
        # state
        self.trend_ = None
        self.seasonal_ = None
        self.resid_ = None
        self.z_scores_ = None
        self.daily_anoms_ = None
        self.weekly_scores_ = None
        self.weekly_anoms_ = None

    def decompose(self, series: np.ndarray):
        """
        Decompose series into trend, seasonal, and residual via STL.
        """
        stl = STL(series, period=self.period, robust=self.stl_robust)
        result = stl.fit()
        self.trend_, self.seasonal_, self.resid_ = (
            result.trend, result.seasonal, result.resid
        )
        return self.trend_, self.seasonal_, self.resid_

    def rolling_mad_zscore(self, residuals: np.ndarray) -> np.ndarray:
        """
        Compute rolling MAD-based z-scores over residuals.
        """
        w = self.rolling_window
        pad = w // 2
        padded = np.pad(residuals, pad, mode='reflect')
        z = np.empty_like(residuals)
        for i in range(len(residuals)):
            win = padded[i:i + w]
            med = np.median(win)
            mad = np.median(np.abs(win - med))
            if mad == 0:
                mad = np.std(win)
            z[i] = 0.6745 * (residuals[i] - med) / mad
        self.z_scores_ = z
        return self.z_scores_

    def flag_point_anomalies(self) -> np.ndarray:
        """
        Return indices of point anomalies where |z| > point_threshold.
        """
        if self.z_scores_ is None:
            raise ValueError("Compute z-scores before flagging anomalies.")
        self.daily_anoms_ = np.where(
            np.abs(self.z_scores_) > self.point_threshold
        )[0]
        return self.daily_anoms_

    def aggregate_window_scores(self) -> np.ndarray:
        """
        Aggregate absolute z-scores over fixed windows (length agg_window).
        """
        if self.z_scores_ is None:
            raise ValueError("Compute z-scores before aggregation.")
        N = len(self.z_scores_)
        num_w = N // self.agg_window
        self.weekly_scores_ = np.array([
            self.agg_func(
                np.abs(self.z_scores_[i*self.agg_window:(i+1)*self.agg_window])
            ) for i in range(num_w)
        ])
        return self.weekly_scores_

    def flag_window_anomalies(self) -> np.ndarray:
        """
        Return window indices (0-based) where aggregated score > percentile cutoff.
        """
        if self.weekly_scores_ is None:
            raise ValueError("Compute window scores before flagging anomalies.")
        thresh = np.percentile(self.weekly_scores_, self.agg_percentile)
        self.weekly_anoms_ = np.where(self.weekly_scores_ > thresh)[0]
        return self.weekly_anoms_
