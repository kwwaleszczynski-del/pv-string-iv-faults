"""Ground-truth labels and scores: the only place where estimator output meets simulator truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .diagnostics import FAULT_CLASSES, NO_DECISION
from .simulator import ChannelScenario, SimulationResult

MIN_IMPACT_PCT = 0.5  # a window counts as faulted only if the fault costs at least this share of its energy


def window_label(sc: ChannelScenario, day_from: int, day_to: int, loss_pct: float) -> str:
    """Ground-truth class of one channel window.

    'Healthy' before onset or for a healthy string; 'Transition' if the onset falls inside the window;
    'Below impact' if the fault is active but costs less than MIN_IMPACT_PCT of the window's energy
    (shading with the sun never behind the obstacle, freshly washed soiling, early PID); otherwise the
    injected fault class.
    """
    if sc.fault_class == "Healthy" or day_to < sc.onset_day:
        return "Healthy"
    if day_from < sc.onset_day:
        return "Transition"
    return sc.fault_class if loss_pct >= MIN_IMPACT_PCT else "Below impact"


def attach_truth(features: pd.DataFrame, result: SimulationResult) -> pd.DataFrame:
    """Add the window label, the true energy loss and the true physical fault size (window means)."""
    tr = result.truth
    doy = tr["doy"].to_numpy()
    sunny = tr["g_poa_wm2"].to_numpy() > 250.0
    by_channel = {sc.channel: (k, sc) for k, sc in enumerate(result.scenarios, start=1)}
    labels, loss, n_bd, rs, soil, rsh = [], [], [], [], [], []
    for _, row in features.iterrows():
        k, sc = by_channel[row["channel"]]
        in_w = (doy >= row["day_from"]) & (doy <= row["day_to"])
        m = in_w & sunny
        p_cf = tr[f"p_counterfactual_mppt_{k}"].to_numpy()[in_w].sum()
        p_true = tr[f"p_true_mppt_{k}"].to_numpy()[in_w].sum()
        loss_pct = 100.0 * (p_cf - p_true) / p_cf if p_cf > 0 else 0.0
        label = window_label(sc, row["day_from"], row["day_to"], loss_pct)
        labels.append(label)
        loss.append(loss_pct)
        n_bd.append(sc.n_bd if label == "Shorted Bypass Diode" else 0)
        rs.append(float(tr[f"rs_fault_mppt_{k}"].to_numpy()[m].mean()) if m.any() else 0.0)
        soil.append(100.0 * float(tr[f"soiling_mppt_{k}"].to_numpy()[m].mean()) if m.any() else 0.0)
        rsh.append(float(tr[f"rsh_pid_mppt_{k}"].to_numpy()[m].mean()) if m.any() else 400.0)
    out = features.copy()
    out["true_class"] = labels
    out["true_loss_pct"] = loss
    out["n_bd_true"] = n_bd
    out["rs_true_ohm"] = rs
    out["soiling_true_pct"] = soil
    out["pid_rsh_true_ohm"] = rsh
    return out


def classification_scores(y_true: list[str], y_pred: list[str]) -> dict:
    """Confusion matrix, per-class precision / recall / F1 and macro-F1 over FAULT_CLASSES.

    'No decision' (too little irradiance in the window) is a column of the confusion matrix and
    counts as a miss for recall; it never counts as a positive for precision.
    """
    columns = FAULT_CLASSES + [NO_DECISION]
    conf = {t: {p: 0 for p in columns} for t in FAULT_CLASSES}
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    metrics, f1s = {}, []
    for c in FAULT_CLASSES:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in FAULT_CLASSES if o != c)
        fn = sum(conf[c][o] for o in columns if o != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if prec + rec else 0.0
        metrics[c] = {
            "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        }
        f1s.append(f1)
    return {"macro_f1": round(float(np.mean(f1s)), 3), "class_metrics": metrics, "confusion_matrix": conf}


def detection_delays(windows: pd.DataFrame, scenarios_by_inverter: dict[int, list[ChannelScenario]]) -> pd.DataFrame:
    """Days from fault onset to the end of the first window that names the right class (NaN if never)."""
    rows = []
    for seed, scenarios in scenarios_by_inverter.items():
        for sc in scenarios:
            if sc.fault_class == "Healthy":
                continue
            w = windows[(windows["seed"] == seed) & (windows["channel"] == sc.channel) & (windows["day_to"] >= sc.onset_day)]
            hit = w[w["predicted_class"] == sc.fault_class]
            delay = float(hit["day_to"].min() - sc.onset_day) if len(hit) else float("nan")
            rows.append({"seed": seed, "channel": sc.channel, "fault_class": sc.fault_class, "onset_day": sc.onset_day, "delay_days": delay})
    return pd.DataFrame(rows)
