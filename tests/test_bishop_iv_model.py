"""Physical properties of the 72-sub-string single-diode and bypass-diode model."""

from __future__ import annotations

import numpy as np
import pytest

from pv_iv_faults.model import (
    InverterStringSpec,
    SubstringFaultState,
    cell_temperature_c,
    find_mppt_operating_point,
    string_mppt,
    sweep_string_iv,
)

SPEC = InverterStringSpec()
HEALTHY = SubstringFaultState()


def test_default_string_spec_geometry():
    assert SPEC.n_substrings == SPEC.n_modules_per_string * SPEC.substrings_per_module == 72
    assert SPEC.v_mp_string_stc_v == pytest.approx(24 * 41.80)


def test_healthy_stc_point_matches_datasheet_within_one_percent():
    v, i, p = find_mppt_operating_point(1000.0, 25.0, HEALTHY, SPEC)
    assert v == pytest.approx(SPEC.v_mp_string_stc_v, rel=0.01)
    assert i == pytest.approx(SPEC.i_mp_stc_a, rel=0.01)
    assert p == pytest.approx(SPEC.n_modules_per_string * SPEC.p_mp_module_stc_w, rel=0.015)


@pytest.mark.parametrize("g, tc", [(250.0, 15.0), (600.0, 35.0), (1000.0, 50.0)])
def test_shorted_bypass_diodes_remove_exactly_n_of_72_sub_strings(g, tc):
    _, i0, p0 = find_mppt_operating_point(g, tc, HEALTHY, SPEC)
    for n in (1, 2, 4):
        _, i_n, p_n = find_mppt_operating_point(g, tc, SubstringFaultState(n_shorted_bypass_diodes=n), SPEC)
        assert 1.0 - p_n / p0 == pytest.approx(n / SPEC.n_substrings, abs=2e-4)
        assert i_n == pytest.approx(i0, abs=1e-3)


def test_series_resistance_drop_grows_with_current_and_stays_below_ohms_law():
    rs = 2.5
    for g in (250.0, 600.0, 950.0):
        v_h, _, _ = find_mppt_operating_point(g, 30.0, HEALTHY, SPEC)
        v_f, i_f, _ = find_mppt_operating_point(g, 30.0, SubstringFaultState(extra_rs_ohm=rs), SPEC)
        # The MPP moves to a slightly lower current and higher internal voltage, so dV < R * I
        assert 0.0 < (v_h - v_f) < rs * i_f
    lo = find_mppt_operating_point(250.0, 30.0, HEALTHY, SPEC)[0] - find_mppt_operating_point(250.0, 30.0, SubstringFaultState(extra_rs_ohm=rs), SPEC)[0]
    hi = find_mppt_operating_point(950.0, 30.0, HEALTHY, SPEC)[0] - find_mppt_operating_point(950.0, 30.0, SubstringFaultState(extra_rs_ohm=rs), SPEC)[0]
    assert hi > 3.0 * lo


def test_pid_shunt_loss_hurts_low_irradiance_more():
    pid = SubstringFaultState(n_pid_modules=4, pid_rsh_module_ohm=35.0)
    deficit = {}
    for g in (150.0, 900.0):
        p_h = find_mppt_operating_point(g, 25.0, HEALTHY, SPEC)[2]
        p_f = find_mppt_operating_point(g, 25.0, pid, SPEC)[2]
        deficit[g] = 1.0 - p_f / p_h
    assert deficit[150.0] > 2.0 * deficit[900.0] > 0.0


def test_partial_shading_has_a_lower_secondary_peak():
    res = sweep_string_iv(900.0, 35.0, SubstringFaultState(n_shaded_substrings=6, shaded_transmission_frac=0.15), SPEC)
    assert res["p_mppt_w"] > res["p_local_w"] > 0.0
    assert abs(res["v_mppt_v"] - res["v_local_v"]) > 40.0


def test_soiling_scales_current_not_voltage():
    v_h, i_h, _ = find_mppt_operating_point(850.0, 32.0, HEALTHY, SPEC)
    v_f, i_f, _ = find_mppt_operating_point(850.0, 32.0, SubstringFaultState(soiling_loss_frac=0.04), SPEC)
    assert 1.0 - i_f / i_h == pytest.approx(0.04, abs=0.003)
    assert abs(1.0 - v_f / v_h) < 0.003


@pytest.mark.parametrize(
    "state",
    [HEALTHY, SubstringFaultState(n_shorted_bypass_diodes=3), SubstringFaultState(extra_rs_ohm=2.0),
     SubstringFaultState(n_pid_modules=3, pid_rsh_module_ohm=40.0), SubstringFaultState(n_shaded_substrings=6, shaded_transmission_frac=0.2)],
)
def test_reported_mpp_is_the_global_maximum_of_the_swept_curve(state):
    res = sweep_string_iv(700.0, 40.0, state, SPEC, n_points=600)
    assert res["p_mppt_w"] >= res["p_w"].max() * (1.0 - 1e-4)


def test_vectorised_call_equals_scalar_calls():
    g = np.array([180.0, 420.0, 760.0, 1010.0])
    tc = np.array([12.0, 28.0, 41.0, 55.0])
    rs = np.array([0.0, 1.0, 2.0, 3.0])
    v, i, p = string_mppt(g, tc, SPEC, extra_rs_ohm=rs, n_shorted=[0, 1, 0, 2])
    for k in range(g.size):
        state = SubstringFaultState(extra_rs_ohm=rs[k], n_shorted_bypass_diodes=[0, 1, 0, 2][k])
        assert (v[k], i[k], p[k]) == pytest.approx(find_mppt_operating_point(g[k], tc[k], state, SPEC), rel=1e-6)


def test_mpp_varies_smoothly_with_irradiance():
    g = np.linspace(500.0, 520.0, 41)
    v, i, _ = string_mppt(g, np.full_like(g, 40.0), SPEC)
    assert np.all(np.diff(i) > 0.0)
    assert np.ptp(np.diff(v)) < 0.02


def test_parallel_string_mppt_topology_8x2():
    v1, i1, p1 = find_mppt_operating_point(800.0, 30.0, HEALTHY, InverterStringSpec(n_mppt=12, strings_per_mppt=1))
    v2, i2, p2 = find_mppt_operating_point(800.0, 30.0, HEALTHY, InverterStringSpec(n_mppt=8, strings_per_mppt=2))
    assert v2 == pytest.approx(v1)
    assert i2 == pytest.approx(2.0 * i1)
    assert p2 == pytest.approx(2.0 * p1)


def test_zero_irradiance_returns_zero_operating_point():
    res = sweep_string_iv(0.0, 15.0, HEALTHY)
    assert res["v_mppt_v"] == res["i_mppt_a"] == res["p_mppt_w"] == 0.0
    assert string_mppt([0.0, 0.5], 15.0, SPEC)[2].tolist() == [0.0, 0.0]


def test_cell_temperature_rises_with_irradiance():
    tc = cell_temperature_c(np.array([0.0, 500.0, 1000.0]), np.array([20.0, 20.0, 20.0]))
    assert tc[0] == 20.0 and tc[2] > tc[1] > tc[0]
