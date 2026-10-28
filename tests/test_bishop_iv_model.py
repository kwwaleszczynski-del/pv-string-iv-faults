"""Unit tests for the 72-sub-string single-diode and Bishop (1988) bypass-diode model."""

from __future__ import annotations

import numpy as np
import pytest

from pv_iv_faults.model import (
    InverterStringSpec,
    SubstringFaultState,
    cell_temperature_c,
    find_mppt_operating_point,
    reference_mppt_iv,
    sweep_string_iv,
)


def test_default_string_spec_geometry():
    spec = InverterStringSpec()
    assert spec.n_mppt == 12
    assert spec.n_modules_per_string == 24
    assert spec.substrings_per_module == 3
    assert spec.n_substrings == 72
    assert pytest.approx(spec.v_mp_string_stc_v, abs=0.5) == 1003.2
    assert pytest.approx(spec.step_voltage_per_shorted_bd_stc_v, abs=0.05) == 14.30


def test_healthy_string_stc_operating_point():
    spec = InverterStringSpec()
    res = sweep_string_iv(1000.0, 25.0, SubstringFaultState(), spec=spec, n_points=800)
    assert pytest.approx(res["v_mppt_v"], abs=2.5) == 1003.2
    assert pytest.approx(res["i_mppt_a"], abs=0.12) == 13.16
    assert pytest.approx(res["p_mppt_w"], rel=0.015) == 24 * 550.0


def test_shorted_bypass_diode_step_voltage_drop():
    spec = InverterStringSpec()
    v0, i0, _ = find_mppt_operating_point(1000.0, 25.0, SubstringFaultState(), spec=spec)
    for n_bd in (1, 2, 4):
        vn, in_a, _ = find_mppt_operating_point(
            1000.0,
            25.0,
            SubstringFaultState(n_shorted_bypass_diodes=n_bd),
            spec=spec,
        )
        dv = v0 - vn
        assert pytest.approx(dv, abs=0.8) == n_bd * 14.30
        assert pytest.approx(in_a, abs=0.08) == i0


def test_shorted_bypass_diode_current_invariance():
    spec = InverterStringSpec()
    fault = SubstringFaultState(n_shorted_bypass_diodes=2)
    dvs = []
    for g in (300.0, 600.0, 950.0):
        v_h, _, _ = find_mppt_operating_point(g, 25.0, SubstringFaultState(), spec=spec)
        v_f, _, _ = find_mppt_operating_point(g, 25.0, fault, spec=spec)
        dvs.append(v_h - v_f)
    assert max(dvs) - min(dvs) < 1.8


def test_series_resistance_proportional_voltage_drop():
    spec = InverterStringSpec()
    rs_extra = 2.5
    fault = SubstringFaultState(extra_rs_ohm=rs_extra)
    v_h_low, i_h_low, _ = find_mppt_operating_point(250.0, 30.0, SubstringFaultState(), spec=spec)
    v_f_low, i_f_low, _ = find_mppt_operating_point(250.0, 30.0, fault, spec=spec)
    v_h_high, i_h_high, _ = find_mppt_operating_point(950.0, 30.0, SubstringFaultState(), spec=spec)
    v_f_high, i_f_high, _ = find_mppt_operating_point(950.0, 30.0, fault, spec=spec)

    dv_low = v_h_low - v_f_low
    dv_high = v_h_high - v_f_high
    assert dv_high > 3.2 * dv_low
    assert pytest.approx(dv_high / i_f_high, rel=0.12) == rs_extra


def test_pid_shunt_degradation_low_irradiance_droop():
    spec = InverterStringSpec()
    fault = SubstringFaultState(n_pid_modules=4, pid_rsh_module_ohm=35.0)
    v_h_low, _, _ = find_mppt_operating_point(200.0, 25.0, SubstringFaultState(), spec=spec)
    v_f_low, _, _ = find_mppt_operating_point(200.0, 25.0, fault, spec=spec)
    v_h_high, _, _ = find_mppt_operating_point(950.0, 25.0, SubstringFaultState(), spec=spec)
    v_f_high, _, _ = find_mppt_operating_point(950.0, 25.0, fault, spec=spec)

    rel_drop_low = 1.0 - (v_f_low / v_h_low)
    rel_drop_high = 1.0 - (v_f_high / v_h_high)
    assert rel_drop_low > rel_drop_high
    assert rel_drop_low > 0.015


def test_partial_shading_creates_global_and_local_maxima():
    spec = InverterStringSpec()
    fault = SubstringFaultState(n_shaded_substrings=6, shaded_transmission_frac=0.35)
    res = sweep_string_iv(900.0, 35.0, fault, spec=spec, n_points=500)
    assert res["p_mppt_w"] > res["p_local_w"]
    assert abs(res["v_mppt_v"] - res["v_local_v"]) > 40.0


def test_uniform_soiling_reduces_current_not_voltage():
    spec = InverterStringSpec()
    fault = SubstringFaultState(soiling_loss_frac=0.045)
    v_h, i_h, _ = find_mppt_operating_point(850.0, 32.0, SubstringFaultState(), spec=spec)
    v_f, i_f, _ = find_mppt_operating_point(850.0, 32.0, fault, spec=spec)
    assert pytest.approx(1.0 - i_f / i_h, abs=0.004) == 0.045
    assert abs(1.0 - v_f / v_h) < 0.004


def test_parallel_string_mppt_topology_8x2():
    spec_1 = InverterStringSpec(n_mppt=12, strings_per_mppt=1)
    spec_2 = InverterStringSpec(n_mppt=8, strings_per_mppt=2)
    v1, i1, p1 = find_mppt_operating_point(800.0, 30.0, SubstringFaultState(), spec=spec_1)
    v2, i2, p2 = find_mppt_operating_point(800.0, 30.0, SubstringFaultState(), spec=spec_2)
    assert pytest.approx(v1, abs=0.5) == v2
    assert pytest.approx(2.0 * i1, abs=0.05) == i2
    assert pytest.approx(2.0 * p1, rel=0.005) == p2


def test_zero_irradiance_returns_zero_operating_point():
    res = sweep_string_iv(0.0, 15.0, SubstringFaultState())
    assert res["v_mppt_v"] == 0.0
    assert res["i_mppt_a"] == 0.0
    assert res["p_mppt_w"] == 0.0


def test_cell_temperature_and_reference_model():
    tc = cell_temperature_c(np.array([0.0, 500.0, 1000.0]), np.array([20.0, 20.0, 20.0]))
    assert tc[0] == 20.0
    assert tc[2] > tc[1] > tc[0]
    v_ref, i_ref = reference_mppt_iv(np.array([0.0, 800.0]), np.array([20.0, 42.0]))
    assert v_ref[0] == 0.0 and i_ref[0] == 0.0
    assert v_ref[1] > 850.0 and i_ref[1] > 9.0
