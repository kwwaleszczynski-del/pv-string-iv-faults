"""Verification tests for fault diagnosis and physical parameter recovery from MPPT data."""

from __future__ import annotations

import pytest

from pv_iv_faults.diagnostics import FAULT_CLASSES, diagnose_mppt_channels, evaluate_fault_verification
from pv_iv_faults.model import InverterStringSpec
from pv_iv_faults.simulator import simulate_annual_mppt_dataset


@pytest.fixture(scope="module")
def annual_run():
    spec = InverterStringSpec()
    df_15m, df_daily = simulate_annual_mppt_dataset(spec=spec, seed=42)
    df_diag = diagnose_mppt_channels(df_15m, spec=spec)
    verification = evaluate_fault_verification(df_15m, df_diag, spec=spec)
    return df_15m, df_daily, df_diag, verification


def test_12_mppt_channels_all_classified_correctly(annual_run):
    _, _, df_diag, _ = annual_run
    assert len(df_diag) == 12
    for _, row in df_diag.iterrows():
        assert row["predicted_class"] == row["true_class"], f"Mismatch on {row['channel']}"


def test_shorted_bypass_diode_count_and_intercept_recovery(annual_run):
    _, _, _, ver = annual_run
    bd_list = ver["bypass_diode_recovery"]
    assert [b["n_bd_true"] for b in bd_list] == [1, 2, 4]
    for b in bd_list:
        assert b["n_bd_est"] == b["n_bd_true"]
        assert b["abs_error_v"] <= 0.40


def test_series_resistance_recovery_within_two_percent(annual_run):
    _, _, _, ver = annual_run
    rs_map = {r["rs_true_ohm"]: r for r in ver["series_resistance_recovery"]}
    assert pytest.approx(rs_map[1.5]["rs_est_ohm"], abs=0.03) == 1.48
    assert pytest.approx(rs_map[3.2]["rs_est_ohm"], abs=0.03) == 3.17
    for r in ver["series_resistance_recovery"]:
        assert abs(r["rel_error_pct"]) < 2.0


def test_pid_shunt_degradation_detection_latency(annual_run):
    _, _, _, ver = annual_run
    pid = ver["pid_verification"]
    assert pid["detection_latency_days"] == 14
    assert pid["low_g_voltage_drop_at_detection_pct"] == 2.2
    assert pid["steady_low_g_voltage_drop_pct"] > 2.2


def test_uniform_soiling_recovery_and_zero_false_voltage_alarms(annual_run):
    _, _, _, ver = annual_run
    soil = ver["soiling_verification"]
    assert soil["soiling_true_pct"] == 4.5
    assert pytest.approx(soil["soiling_est_pct"], abs=0.15) == 4.4
    assert soil["false_voltage_alarm_rate_pct"] == 0.0


def test_six_class_macro_f1_score(annual_run):
    _, _, _, ver = annual_run
    assert set(ver["class_metrics"].keys()) == set(FAULT_CLASSES)
    assert pytest.approx(ver["macro_f1"], abs=0.015) == 0.968
