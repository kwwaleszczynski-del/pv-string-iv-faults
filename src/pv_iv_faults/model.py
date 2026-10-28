"""Physics-based 72-sub-string one-diode and bypass-diode model (Bishop 1988)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InverterStringSpec:
    """Electrical specification of a commercial string inverter and its mono-PERC strings.

    Default configuration models a 12-MPPT single-string commercial inverter (or 8 MPPT x 2
    parallel strings), with 24 x 550 Wp half-cut mono-PERC modules per string and 3 sub-strings
    (bypass diodes) per module = 72 sub-strings per string.
    """

    n_mppt: int = 12
    strings_per_mppt: int = 1
    n_modules_per_string: int = 24
    substrings_per_module: int = 3

    # Module STC parameters (G = 1000 W/m2, Tc = 25 degC)
    p_mp_module_stc_w: float = 550.0
    v_mp_module_stc_v: float = 41.80
    i_mp_stc_a: float = 13.16
    v_oc_module_stc_v: float = 49.80
    i_sc_stc_a: float = 13.98

    # Temperature coefficients
    alpha_isc_pct_per_k: float = 0.048
    beta_voc_pct_per_k: float = -0.270
    beta_vmp_pct_per_k: float = -0.310

    # Sub-string 5-parameter + Bishop (1988) breakdown + parallel Schottky bypass diode
    cells_per_substring: int = 24
    diode_ideality: float = 1.125
    rs_substring_ohm: float = 0.0445
    rsh_module_healthy_ohm: float = 400.0
    v_bypass_drop_v: float = 0.55
    v_shorted_diode_drop_v: float = 0.367  # yields 13.933 + 0.367 = 14.30 V step per shorted sub-string at STC
    bishop_a: float = 0.002
    bishop_m: float = 3.2
    bishop_vbr_v: float = -18.5

    # Thermal model (NOCT / PVsyst-style)
    u_c_w_m2k: float = 29.0

    @property
    def n_substrings(self) -> int:
        return self.n_modules_per_string * self.substrings_per_module

    @property
    def v_mp_string_stc_v(self) -> float:
        return self.n_modules_per_string * self.v_mp_module_stc_v

    @property
    def v_oc_string_stc_v(self) -> float:
        return self.n_modules_per_string * self.v_oc_module_stc_v

    @property
    def v_mp_substring_stc_v(self) -> float:
        return self.v_mp_module_stc_v / self.substrings_per_module

    @property
    def v_oc_substring_stc_v(self) -> float:
        return self.v_oc_module_stc_v / self.substrings_per_module

    @property
    def rsh_substring_healthy_ohm(self) -> float:
        return self.rsh_module_healthy_ohm / self.substrings_per_module

    @property
    def step_voltage_per_shorted_bd_stc_v(self) -> float:
        return self.v_mp_substring_stc_v + self.v_shorted_diode_drop_v


@dataclass(frozen=True)
class SubstringFaultState:
    """Physical fault state applied to an MPPT channel's string(s)."""

    n_shorted_bypass_diodes: int = 0
    extra_rs_ohm: float = 0.0
    n_pid_modules: int = 0
    pid_rsh_module_ohm: float = 400.0
    pid_voc_loss_frac: float = 0.0
    n_shaded_substrings: int = 0
    shaded_transmission_frac: float = 1.0
    soiling_loss_frac: float = 0.0


def cell_temperature_c(g_poa_wm2: float | np.ndarray, t_amb_c: float | np.ndarray, spec: InverterStringSpec = InverterStringSpec()) -> float | np.ndarray:
    """Compute PV cell temperature (degC) from plane-of-array irradiance and ambient temperature."""
    return t_amb_c + (g_poa_wm2 / 1000.0) * (800.0 / spec.u_c_w_m2k)


def _substring_voltage_given_current(
    currents_a: np.ndarray,
    g_eff_wm2: float,
    t_cell_c: float,
    rsh_sub_ohm: float,
    spec: InverterStringSpec,
    voc_scale: float = 1.0,
) -> np.ndarray:
    """Evaluate sub-string terminal voltage V_sub(I) using the 5-parameter + Bishop (1988) + bypass diode model."""
    if g_eff_wm2 <= 1.0:
        return np.zeros_like(currents_a, dtype=float)

    dt = t_cell_c - 25.0
    k_b = 1.380649e-23
    q_e = 1.602176634e-19
    t_k = t_cell_c + 273.15
    t_stc_k = 298.15

    a_stc = spec.cells_per_substring * spec.diode_ideality * (k_b * t_stc_k / q_e)
    a_t = a_stc * (t_k / t_stc_k)

    # Photocurrent scaled with irradiance and temperature
    i_ph_stc = spec.i_sc_stc_a * (1.0 + spec.rs_substring_ohm / spec.rsh_substring_healthy_ohm)
    i_ph = i_ph_stc * (g_eff_wm2 / 1000.0) * (1.0 + (spec.alpha_isc_pct_per_k / 100.0) * dt)

    # Open-circuit voltage of one sub-string at (G, Tc)
    v_oc_sub_stc = spec.v_oc_substring_stc_v
    v_oc_sub = (
        v_oc_sub_stc * (1.0 + (spec.beta_voc_pct_per_k / 100.0) * dt)
        + a_t * np.log(max(g_eff_wm2 / 1000.0, 1e-4))
    ) * voc_scale
    v_oc_sub = max(v_oc_sub, 2.0)

    # Saturation current I_0 anchored to match V_oc_sub at healthy R_sh
    i_sh_oc = v_oc_sub / max(spec.rsh_substring_healthy_ohm, 10.0)
    i_0 = max(i_ph - i_sh_oc, 1e-12) / np.expm1(v_oc_sub / a_t)

    # Junction voltage grid V_d spanning reverse bias (bypass clamping + Bishop breakdown) to above V_oc
    v_d = np.linspace(-1.4, v_oc_sub + 0.35, 650)

    # Forward diode current
    exp_arg = np.clip(v_d / a_t, -40.0, 60.0)
    i_diode = i_0 * np.expm1(exp_arg)

    # Bishop (1988) reverse-bias avalanche multiplier on the shunt term
    ratio_br = np.clip(v_d / spec.bishop_vbr_v, -5.0, 0.98)
    bishop_mult = 1.0 + spec.bishop_a * np.power(1.0 - ratio_br, -spec.bishop_m)
    i_shunt = (v_d / max(rsh_sub_ohm, 0.25)) * bishop_mult

    i_cell = i_ph - i_diode - i_shunt
    v_sub = v_d - i_cell * spec.rs_substring_ohm

    # Parallel Schottky/Si bypass diode conducting when V_sub < 0
    # Calibrated so I_bp = I_mp_stc at V_sub = -spec.v_bypass_drop_v (-0.55 V)
    a_bp = 0.042
    i0_bp = spec.i_mp_stc_a / np.expm1(spec.v_bypass_drop_v / a_bp)
    bp_arg = np.clip(-v_sub / a_bp, -40.0, 50.0)
    i_bp = i0_bp * np.expm1(bp_arg)

    i_total = i_cell + i_bp

    # Since i_total decreases monotonically as v_d (and v_sub) increases, invert by interpolation
    order = np.argsort(i_total)
    i_sorted = i_total[order]
    v_sorted = v_sub[order]
    v_interp = np.interp(currents_a, i_sorted, v_sorted, left=v_sorted[0], right=v_sorted[-1])
    return v_interp


def sweep_string_iv(
    g_poa_wm2: float,
    t_cell_c: float,
    fault: SubstringFaultState = SubstringFaultState(),
    spec: InverterStringSpec = InverterStringSpec(),
    n_points: int = 400,
) -> dict[str, np.ndarray | float]:
    """Sweep the full I-V and P-V characteristic of a 72-sub-string PV string (or parallel string pair).

    Returns a dictionary containing arrays `v_v`, `i_a`, `p_w`, plus the global MPPT operating point
    (`v_mppt_v`, `i_mppt_a`, `p_mppt_w`) and any secondary local maximum (`v_local_v`, `p_local_w`).
    """
    n_total = spec.n_substrings
    n_short = int(np.clip(fault.n_shorted_bypass_diodes, 0, n_total - 1))
    n_pid = int(np.clip(fault.n_pid_modules * spec.substrings_per_module, 0, n_total - n_short))
    n_shaded = int(np.clip(fault.n_shaded_substrings, 0, n_total - n_short - n_pid))
    n_healthy = n_total - n_short - n_pid - n_shaded

    g_clean = max(float(g_poa_wm2), 0.0)
    g_unshaded = g_clean * (1.0 - np.clip(fault.soiling_loss_frac, 0.0, 0.95))
    g_shaded = g_unshaded * np.clip(fault.shaded_transmission_frac, 0.02, 1.0)

    if g_unshaded <= 1.0:
        zeros = np.zeros(n_points, dtype=float)
        return {
            "v_v": zeros,
            "i_a": zeros,
            "p_w": zeros,
            "v_mppt_v": 0.0,
            "i_mppt_a": 0.0,
            "p_mppt_w": 0.0,
            "v_local_v": 0.0,
            "p_local_w": 0.0,
        }

    dt = t_cell_c - 25.0
    i_sc_max = spec.i_sc_stc_a * (g_unshaded / 1000.0) * (1.0 + (spec.alpha_isc_pct_per_k / 100.0) * dt) * 1.015
    i_string = np.linspace(0.0, max(i_sc_max, 0.05), n_points)

    v_healthy = _substring_voltage_given_current(
        i_string,
        g_eff_wm2=g_unshaded,
        t_cell_c=t_cell_c,
        rsh_sub_ohm=spec.rsh_substring_healthy_ohm,
        spec=spec,
    )

    v_total = n_healthy * v_healthy

    if n_short > 0:
        # Shorted bypass diode conducts with forward contact drop -v_shorted_diode_drop_v
        v_total = v_total - n_short * spec.v_shorted_diode_drop_v

    if n_pid > 0:
        rsh_pid_sub = max(fault.pid_rsh_module_ohm / spec.substrings_per_module, 0.5)
        # PID-affected negative-end modules suffer shunt leakage and junction recombination loss
        # that depresses voltage strongly at low irradiance (G < 300 W/m2)
        low_g_factor = np.clip((1000.0 - g_unshaded) / 800.0, 0.15, 1.0)
        shunt_severity = np.clip((spec.rsh_module_healthy_ohm - fault.pid_rsh_module_ohm) / spec.rsh_module_healthy_ohm, 0.0, 1.0)
        voc_scale = 1.0 - (0.045 * shunt_severity + fault.pid_voc_loss_frac) * (0.35 + 0.95 * low_g_factor)
        v_pid = _substring_voltage_given_current(
            i_string,
            g_eff_wm2=g_unshaded,
            t_cell_c=t_cell_c,
            rsh_sub_ohm=rsh_pid_sub,
            spec=spec,
            voc_scale=voc_scale,
        )
        v_total = v_total + n_pid * v_pid

    if n_shaded > 0:
        v_shaded = _substring_voltage_given_current(
            i_string,
            g_eff_wm2=g_shaded,
            t_cell_c=t_cell_c,
            rsh_sub_ohm=spec.rsh_substring_healthy_ohm,
            spec=spec,
        )
        v_total = v_total + n_shaded * v_shaded

    # Subtract extra series resistance voltage drop (corroded MC4 connector / solder joint)
    v_total = v_total - i_string * max(fault.extra_rs_ohm, 0.0)

    # Keep only first-quadrant operating points (V >= 0)
    valid = v_total >= 0.0
    if not np.any(valid):
        valid[0] = True
    v_pos = np.maximum(v_total, 0.0)
    i_mppt_channel = i_string * spec.strings_per_mppt
    p_pos = v_pos * i_mppt_channel

    # Sort in ascending voltage order for clean plotting and peak detection
    v_ord = np.argsort(v_pos)
    v_curve = v_pos[v_ord]
    i_curve = i_mppt_channel[v_ord]
    p_curve = p_pos[v_ord]

    idx_global = int(np.argmax(p_curve))
    v_mppt = float(v_curve[idx_global])
    i_mppt = float(i_curve[idx_global])
    p_mppt = float(p_curve[idx_global])

    # Detect secondary local peak (when partial shading creates two distinct power maxima)
    v_local = v_mppt
    p_local = p_mppt
    if n_shaded > 0:
        # Find local maximum in the unbypassed high-voltage branch beyond the bypass knee
        high_branch = v_curve > (v_mppt + 155.0)
        if np.any(high_branch):
            idx_sec = int(np.argmax(np.where(high_branch, p_curve, -1.0)))
            if p_curve[idx_sec] > 0.15 * p_mppt:
                v_local = float(v_curve[idx_sec])
                p_local = float(p_curve[idx_sec])

    return {
        "v_v": v_curve,
        "i_a": i_curve,
        "p_w": p_curve,
        "v_mppt_v": v_mppt,
        "i_mppt_a": i_mppt,
        "p_mppt_w": p_mppt,
        "v_local_v": v_local,
        "p_local_w": p_local,
    }


def find_mppt_operating_point(
    g_poa_wm2: float,
    t_cell_c: float,
    fault: SubstringFaultState = SubstringFaultState(),
    spec: InverterStringSpec = InverterStringSpec(),
) -> tuple[float, float, float]:
    """Return (V_mppt, I_mppt, P_mppt) for a single irradiance and cell temperature point."""
    res = sweep_string_iv(g_poa_wm2=g_poa_wm2, t_cell_c=t_cell_c, fault=fault, spec=spec, n_points=320)
    return float(res["v_mppt_v"]), float(res["i_mppt_a"]), float(res["p_mppt_w"])


def reference_mppt_iv(
    g_poa_wm2: float | np.ndarray,
    t_cell_c: float | np.ndarray,
    spec: InverterStringSpec = InverterStringSpec(),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute clear-sky temperature-corrected reference (V_ref, I_ref) for healthy strings.

    Calibrated directly to the healthy 72-sub-string single-diode model across (G_poa, T_c).
    """
    g_arr = np.asarray(g_poa_wm2, dtype=float)
    t_arr = np.asarray(t_cell_c, dtype=float)
    dt = t_arr - 25.0

    k_b = 1.380649e-23
    q_e = 1.602176634e-19
    a_t = spec.n_substrings * spec.cells_per_substring * spec.diode_ideality * (k_b * (t_arr + 273.15) / q_e)

    g_ratio = np.clip(g_arr / 1000.0, 1e-4, 1.5)
    # Single-diode healthy MPPT voltage & current tracking with irradiance and cell temperature
    i_ref = (
        spec.i_mp_stc_a
        * spec.strings_per_mppt
        * g_ratio
        * (1.0 + (spec.alpha_isc_pct_per_k / 100.0) * dt)
    )
    v_ref = (
        spec.v_mp_string_stc_v * (1.0 + (spec.beta_vmp_pct_per_k / 100.0) * dt)
        + 0.94 * a_t * np.log(g_ratio)
        - spec.n_substrings * spec.rs_substring_ohm * (i_ref / spec.strings_per_mppt - spec.i_mp_stc_a)
    )
    mask = g_arr > 15.0
    return np.where(mask, v_ref, 0.0), np.where(mask, i_ref, 0.0)
