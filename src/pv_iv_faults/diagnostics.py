"""Truth-blind fault estimators for PV strings from 15-minute MPPT telemetry.

Inputs are only what a SCADA historian and the plant documentation provide: timestamps, measured
plane-of-array irradiance, per-MPPT voltage and current, the site coordinates (for solar position)
and the module datasheet (sub-strings per string). This module does not import the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import InverterStringSpec
from .solar import solar_position

FAULT_CLASSES: list[str] = [
    "Healthy",
    "Shorted Bypass Diode",
    "High Series Resistance R_s",
    "PID / Low R_sh",
    "Partial Shading",
    "Uniform Soiling",
]

NO_DECISION = "No decision"

# Feature that drives each detector, in the order the rules are applied
DETECTORS: list[tuple[str, str]] = [
    ("Partial Shading", "shade_score"),
    ("PID / Low R_sh", "pid_droop"),
    ("High Series Resistance R_s", "dv_slope_ohm"),
    ("Shorted Bypass Diode", "v_def_hi"),
    ("Uniform Soiling", "i_def_hi"),
]

WINDOW_DAYS = 14
G_HIGH_WM2 = 450.0
G_LOW_WM2 = (100.0, 250.0)
G_FIT_WM2 = 250.0
MIN_FIT_SAMPLES = 40
MIN_BIN_SAMPLES = 10
MIN_SUN_BIN_SAMPLES = 10
SHADE_G_MIN_WM2 = 150.0
G_DETREND_BIN_WM2 = 50.0


@dataclass
class Samples:
    """Per-timestamp arrays shared by all channels of one inverter."""

    doy: np.ndarray
    g: np.ndarray
    el: np.ndarray
    az: np.ndarray
    v: np.ndarray  # (n, k) channel voltages
    i: np.ndarray  # (n, k) channel currents
    v_peer: np.ndarray
    i_peer: np.ndarray
    p_peer: np.ndarray
    v_def: np.ndarray  # (n, k) 1 - V_k / V_peer
    i_def: np.ndarray  # (n, k) 1 - I_k / I_peer
    p_def: np.ndarray  # (n, k) 1 - P_k / P_peer


def peer_reference(v: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-timestamp peer (V, I, P): medians over the upper half of the channels ranked by power.

    Every string fault lowers power, while some raise voltage or current on their own (a PID-affected
    string can sit at a higher MPP voltage and lower current). Ranking by power therefore keeps
    degraded channels out of the reference as long as fewer than half of them lose power at once.
    """
    p = v * i
    k = p.shape[1]
    top = np.argsort(-p, axis=1)[:, : (k + 1) // 2]
    return (
        np.median(np.take_along_axis(v, top, axis=1), axis=1),
        np.median(np.take_along_axis(i, top, axis=1), axis=1),
        np.median(np.take_along_axis(p, top, axis=1), axis=1),
    )


def prepare_samples(telemetry: pd.DataFrame, lat_deg: float, lon_deg: float) -> Samples:
    """Build peer references and solar geometry from telemetry columns `v_mppt_k`, `i_mppt_k`."""
    k = sum(1 for c in telemetry.columns if c.startswith("v_mppt_"))
    v = telemetry[[f"v_mppt_{j}" for j in range(1, k + 1)]].to_numpy(dtype=float)
    i = telemetry[[f"i_mppt_{j}" for j in range(1, k + 1)]].to_numpy(dtype=float)
    ts = pd.DatetimeIndex(telemetry["timestamp"])
    el, az = solar_position(ts, lat_deg, lon_deg)
    v_peer, i_peer, p_peer = peer_reference(v, i)

    def deficit(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        ratio = np.divide(x, ref[:, None], out=np.full_like(x, np.nan), where=ref[:, None] > 0.0)
        return 1.0 - ratio

    return Samples(
        doy=ts.day_of_year.to_numpy(),
        g=telemetry["g_poa_wm2"].to_numpy(dtype=float),
        el=el,
        az=az,
        v=v,
        i=i,
        v_peer=v_peer,
        i_peer=i_peer,
        p_peer=p_peer,
        v_def=deficit(v, v_peer),
        i_def=deficit(i, i_peer),
        p_def=deficit(v * i, p_peer),
    )


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares line refitted once without residuals beyond 4 scaled MADs."""
    slope, intercept = np.polyfit(x, y, 1)
    res = y - (slope * x + intercept)
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    keep = np.abs(res) <= 4.0 * mad + 0.5
    if keep.sum() >= MIN_FIT_SAMPLES // 2:
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return float(slope), float(intercept)


def _shade_score(p_def: np.ndarray, g: np.ndarray, el: np.ndarray, az: np.ndarray) -> tuple[float, float, float]:
    """Largest sun-position-specific power deficit after removing the irradiance-explained part.

    The deficit is detrended by its median in 100 W/m2 irradiance bins, then binned by solar
    position (10 deg azimuth x 5 deg elevation). Returns (score, azimuth, elevation) of the worst bin.
    """
    g_bin = np.floor(g / G_DETREND_BIN_WM2).astype(int)
    trend = pd.Series(p_def).groupby(g_bin).transform("median").to_numpy()
    resid = p_def - trend
    key = pd.DataFrame({"az": np.floor(az / 10.0) * 10.0 + 5.0, "el": np.floor(el / 5.0) * 5.0 + 2.5, "r": resid})
    grouped = key.groupby(["az", "el"])["r"]
    counts = grouped.size()
    q75 = grouped.quantile(0.75)[counts >= MIN_SUN_BIN_SAMPLES]
    if q75.empty:
        return 0.0, float("nan"), float("nan")
    (az_c, el_c), score = q75.idxmax(), float(q75.max())
    return score, float(az_c), float(el_c)


def channel_features(s: Samples, k: int, day_from: int, day_to: int) -> dict:
    """Window features of channel index k (0-based) over days [day_from, day_to]."""
    v_k, i_k = s.v[:, k], s.i[:, k]
    base = (
        (s.doy >= day_from) & (s.doy <= day_to) & (s.g > 50.0) & (s.el > 3.0)
        & (s.v_peer > 100.0) & (s.i_peer > 0.2)
    )
    fit = base & (s.g > G_FIT_WM2)
    hi = base & (s.g >= G_HIGH_WM2)
    lo = base & (s.g >= G_LOW_WM2[0]) & (s.g <= G_LOW_WM2[1])
    out = {"n_fit": int(fit.sum()), "n_hi": int(hi.sum()), "n_lo": int(lo.sum())}
    out["sufficient"] = bool(out["n_fit"] >= MIN_FIT_SAMPLES and out["n_hi"] >= MIN_BIN_SAMPLES and out["n_lo"] >= MIN_BIN_SAMPLES)
    if not out["sufficient"]:
        return out

    v_def, i_def, p_def = s.v_def[:, k], s.i_def[:, k], s.p_def[:, k]
    slope, intercept = _robust_line(i_k[fit], (s.v_peer - v_k)[fit])
    out.update(
        dv_slope_ohm=slope,
        dv_intercept_v=intercept,
        v_def_hi=float(np.median(v_def[hi])),
        v_def_lo=float(np.median(v_def[lo])),
        i_def_hi=float(np.median(i_def[hi])),
        i_def_lo=float(np.median(i_def[lo])),
        p_def_hi=float(np.median(p_def[hi])),
        p_def_lo=float(np.median(p_def[lo])),
        v_peer_hi=float(np.median(s.v_peer[hi])),
    )
    out["pid_droop"] = out["p_def_lo"] - out["p_def_hi"]
    shade = base & (s.g > SHADE_G_MIN_WM2)
    out["shade_score"], out["shade_az_deg"], out["shade_el_deg"] = _shade_score(p_def[shade], s.g[shade], s.el[shade], s.az[shade])
    return out


def window_features(s: Samples, window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Features for every channel and every consecutive `window_days` window of the year."""
    rows = []
    n_windows = 365 // window_days
    for k in range(s.v.shape[1]):
        for w in range(n_windows):
            d0, d1 = w * window_days + 1, (w + 1) * window_days
            rows.append({"channel": f"MPPT {k + 1}", "window": w, "day_from": d0, "day_to": d1, **channel_features(s, k, d0, d1)})
    return pd.DataFrame(rows)


def fit_thresholds(healthy_features: pd.DataFrame, quantile: float = 0.99) -> dict[str, float]:
    """Detector thresholds = the `quantile` of each feature over fault-free windows (tuning data only)."""
    ok = healthy_features[healthy_features["sufficient"]]
    return {feat: float(np.quantile(ok[feat].to_numpy(dtype=float), quantile)) for _, feat in DETECTORS}


def classify(features: pd.DataFrame, thresholds: dict[str, float], spec: InverterStringSpec = InverterStringSpec()) -> pd.DataFrame:
    """Apply the ordered detector rules and the physical parameter estimators to feature rows."""
    out = features.copy()
    pred, n_bd, rs, soil = [], [], [], []
    for _, row in out.iterrows():
        label = "Healthy" if row["sufficient"] else NO_DECISION
        if row["sufficient"]:
            for cls, feat in DETECTORS:
                if row[feat] > thresholds[feat]:
                    label = cls
                    break
        pred.append(label)
        # A shorted bypass diode removes one of N_sub series sub-strings: v_def = n / N_sub
        n_bd.append(max(1, int(round(row["v_def_hi"] * spec.n_substrings))) if label == "Shorted Bypass Diode" else 0)
        rs.append(round(float(row["dv_slope_ohm"]), 3) if label == "High Series Resistance R_s" else 0.0)
        soil.append(round(100.0 * float(row["i_def_hi"]), 2) if label == "Uniform Soiling" else 0.0)
    out["predicted_class"] = pred
    out["n_bd_est"] = n_bd
    out["rs_est_ohm"] = rs
    out["soiling_est_pct"] = soil
    return out


def daily_tracking(s: Samples) -> pd.DataFrame:
    """Daily medians (G > 150 W/m2) of voltage ratio, current ratio and voltage deficit per channel."""
    m = (s.g > 150.0) & (s.v_peer > 100.0) & (s.i_peer > 0.2)
    frame = {"doy": s.doy[m]}
    for k in range(s.v.shape[1]):
        slug = f"mppt_{k + 1}"
        frame[f"v_norm_{slug}"] = 1.0 - s.v_def[m, k]
        frame[f"i_norm_{slug}"] = 1.0 - s.i_def[m, k]
        frame[f"dv_med_{slug}"] = s.v_peer[m] - s.v[m, k]
    return pd.DataFrame(frame).groupby("doy", sort=True).median().reset_index()


def estimated_loss_mwh(s: Samples) -> np.ndarray:
    """Annual energy deficit of each channel against the peer reference power (signed, MWh)."""
    p = s.v * s.i
    return (s.p_peer[:, None] - p).sum(axis=0) * 0.25 / 1e6
