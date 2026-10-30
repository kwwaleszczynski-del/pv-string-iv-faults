"""Properties of the truth-blind estimator on small synthetic inverters (not the reported data sets)."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from pv_iv_faults import diagnostics
from pv_iv_faults.diagnostics import (
    DETECTORS,
    FAULT_CLASSES,
    NO_DECISION,
    classify,
    fit_thresholds,
    peer_reference,
    prepare_samples,
    window_features,
)
from pv_iv_faults.simulator import FAULT_FAMILIES, ChannelScenario, simulate_inverter

HEALTHY_6 = [ChannelScenario(f"MPPT {k}") for k in range(1, 7)]
FAULTED_6 = [
    ChannelScenario("MPPT 1", "Shorted Bypass Diode", n_bd=3),
    ChannelScenario("MPPT 2", "High Series Resistance R_s", rs_final_ohm=3.0),
    *[ChannelScenario(f"MPPT {k}") for k in range(3, 7)],
]


@pytest.fixture(scope="module")
def runs():
    """Thresholds from a fault-free inverter, applied to a second inverter with a different seed."""
    ref = simulate_inverter(HEALTHY_6, seed=501)
    test = simulate_inverter(FAULTED_6, seed=502)
    f_ref = window_features(prepare_samples(ref.telemetry, ref.site.lat_deg, ref.site.lon_deg))
    thresholds = fit_thresholds(f_ref)
    pred = classify(window_features(prepare_samples(test.telemetry, test.site.lat_deg, test.site.lon_deg)), thresholds)
    return test, thresholds, pred


def test_estimator_module_does_not_import_the_simulator():
    tree = ast.parse(inspect.getsource(diagnostics))
    imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any("simulator" in m or "evaluation" in m for m in imported)


def test_telemetry_holds_only_scada_columns(runs):
    test, _, _ = runs
    expected = {"timestamp", "g_poa_wm2", "t_amb_c"} | {f"{q}_mppt_{k}" for q in "vi" for k in range(1, 7)}
    assert set(test.telemetry.columns) == expected


def test_class_names_agree_between_generator_and_estimator():
    assert FAULT_CLASSES == ["Healthy", *FAULT_FAMILIES]


def test_peer_reference_is_invariant_to_channel_order():
    rng = np.random.default_rng(0)
    v = rng.uniform(800.0, 1000.0, (50, 7))
    i = rng.uniform(5.0, 12.0, (50, 7))
    perm = rng.permutation(7)
    for a, b in zip(peer_reference(v, i), peer_reference(v[:, perm], i[:, perm])):
        np.testing.assert_allclose(a, b)


def test_peer_reference_ignores_a_minority_of_power_losers():
    v = np.full((10, 6), 900.0)
    i = np.full((10, 6), 10.0)
    v[:, :2] = 850.0  # two channels lose voltage (and power)
    i[:, 2] = 9.0  # one channel loses current (and power)
    v_peer, i_peer, p_peer = peer_reference(v, i)
    assert np.all(v_peer == 900.0) and np.all(i_peer == 10.0) and np.all(p_peer == 9000.0)


def test_thresholds_are_the_requested_quantile_of_healthy_features():
    rows = pd.DataFrame({"sufficient": [True] * 101, **{feat: np.linspace(0.0, 1.0, 101) for _, feat in DETECTORS}})
    th = fit_thresholds(rows, quantile=0.9)
    assert all(v == pytest.approx(0.9) for v in th.values())


def test_rules_follow_detector_order_and_abstain_without_data():
    th = {feat: 1.0 for _, feat in DETECTORS}
    base = {"sufficient": True, "v_def_hi": 0.03, "dv_slope_ohm": 0.0, "i_def_hi": 0.0, "pid_droop": 0.0, "shade_score": 0.0}
    rows = pd.DataFrame([
        {**base},
        {**base, "v_def_hi": 2.0},
        {**base, "v_def_hi": 2.0, "dv_slope_ohm": 2.0},
        {**base, "i_def_hi": 2.0, "shade_score": 2.0},
        {"sufficient": False},
    ])
    assert classify(rows, th)["predicted_class"].tolist() == [
        "Healthy", "Shorted Bypass Diode", "High Series Resistance R_s", "Partial Shading", NO_DECISION,
    ]


def test_fault_free_channels_rarely_alarm_on_an_unseen_inverter(runs):
    _, _, pred = runs
    healthy = pred[pred["channel"].isin([f"MPPT {k}" for k in range(3, 7)]) & (pred["predicted_class"] != NO_DECISION)]
    assert len(healthy) > 50
    assert (healthy["predicted_class"] != "Healthy").mean() <= 0.12


def test_shorted_diode_count_follows_from_the_voltage_ratio(runs):
    _, _, pred = runs
    ch = pred[(pred["channel"] == "MPPT 1") & (pred["predicted_class"] != NO_DECISION)]
    hits = ch[ch["predicted_class"] == "Shorted Bypass Diode"]
    assert len(hits) >= 0.9 * len(ch)
    assert (hits["n_bd_est"] == 3).mean() >= 0.9


def test_series_resistance_slope_is_below_the_injected_resistance(runs):
    _, _, pred = runs
    ch = pred[(pred["channel"] == "MPPT 2") & (pred["predicted_class"] == "High Series Resistance R_s")]
    assert len(ch) >= 10
    slope = ch["rs_est_ohm"].median()
    assert 0.6 * 3.0 < slope < 1.05 * 3.0
