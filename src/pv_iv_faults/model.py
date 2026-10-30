"""Sub-string single-diode model with Bishop (1988) reverse-bias term and bypass diodes.

Every voltage and current the simulator produces comes out of this module: a string is a series
connection of 72 sub-strings, each solved on its own junction-voltage grid and combined at a common
string current. The maximum power point is searched numerically on that combined curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_B = 1.380649e-23
Q_E = 1.602176634e-19
N_JUNCTION_GRID = 400


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
    bishop_a: float = 0.002
    bishop_m: float = 3.2
    bishop_vbr_v: float = -18.5

    # Thermal model (NOCT-style U-value)
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


@dataclass(frozen=True)
class SubstringFaultState:
    """Physical fault state applied to an MPPT channel's string(s).

    A shorted bypass diode is modelled as an ideal short across its sub-string (0 V at any current).
    `shaded_transmission_frac` is the irradiance on the shaded sub-strings relative to the rest.
    """

    n_shorted_bypass_diodes: int = 0
    extra_rs_ohm: float = 0.0
    n_pid_modules: int = 0
    pid_rsh_module_ohm: float = 400.0
    n_shaded_substrings: int = 0
    shaded_transmission_frac: float = 1.0
    soiling_loss_frac: float = 0.0


def cell_temperature_c(g_poa_wm2: float | np.ndarray, t_amb_c: float | np.ndarray, spec: InverterStringSpec = InverterStringSpec()) -> float | np.ndarray:
    """Compute PV cell temperature (degC) from plane-of-array irradiance and ambient temperature."""
    return t_amb_c + (g_poa_wm2 / 1000.0) * (800.0 / spec.u_c_w_m2k)


A_BYPASS_V = 0.042  # bypass diode thermal voltage x ideality


class _Substring:
    """Sub-string single-diode parameters for one operating condition per array element.

    5-parameter single-diode equation, Bishop (1988) avalanche multiplier on the shunt term and a
    parallel bypass diode, written as terminal current versus junction voltage `v_d`.
    """

    def __init__(self, g_eff_wm2, t_cell_c, rsh_sub_ohm, spec: InverterStringSpec):
        g = np.maximum(np.asarray(g_eff_wm2, dtype=float), 1.0)
        t = np.asarray(t_cell_c, dtype=float)
        dt = t - 25.0
        self.spec = spec
        self.rsh = np.maximum(np.asarray(rsh_sub_ohm, dtype=float), 0.25)
        self.a_t = spec.cells_per_substring * spec.diode_ideality * K_B * (t + 273.15) / Q_E

        # Photocurrent scaled with irradiance and temperature
        i_ph_stc = spec.i_sc_stc_a * (1.0 + spec.rs_substring_ohm / spec.rsh_substring_healthy_ohm)
        self.i_ph = i_ph_stc * (g / 1000.0) * (1.0 + (spec.alpha_isc_pct_per_k / 100.0) * dt)

        # Open-circuit voltage of a healthy sub-string at (G, Tc) anchors the diode saturation current
        v_oc = spec.v_oc_substring_stc_v * (1.0 + (spec.beta_voc_pct_per_k / 100.0) * dt) + self.a_t * np.log(
            np.maximum(g / 1000.0, 1e-4)
        )
        self.v_oc = np.maximum(v_oc, 2.0)
        self.i_0 = np.maximum(self.i_ph - self.v_oc / spec.rsh_substring_healthy_ohm, 1e-12) / np.expm1(self.v_oc / self.a_t)

        # Parallel bypass diode, calibrated so I_bp = I_mp_stc at V_sub = -v_bypass_drop_v
        self.i0_bp = spec.i_mp_stc_a / np.expm1(spec.v_bypass_drop_v / A_BYPASS_V)

    def current(self, v_d: np.ndarray, col: bool = False, derivs: bool = True):
        """Return (I_total, V_sub[, dI_total/dv_d]) at junction voltage v_d."""
        sp = self.spec
        a_t, i_0, i_ph, rsh = self.a_t, self.i_0, self.i_ph, self.rsh
        if col:
            a_t, i_0, i_ph, rsh = a_t[:, None], i_0[:, None], i_ph[:, None], rsh[:, None]
        e = np.exp(np.clip(v_d / a_t, -40.0, 60.0))
        q = 1.0 / (1.0 - np.clip(v_d / sp.bishop_vbr_v, -5.0, 0.98))
        q_m = q**sp.bishop_m
        mult = 1.0 + sp.bishop_a * q_m

        i_cell = i_ph - i_0 * (e - 1.0) - (v_d / rsh) * mult
        v_sub = v_d - i_cell * sp.rs_substring_ohm
        e_bp = np.exp(np.clip(-v_sub / A_BYPASS_V, -40.0, 50.0))
        i_tot = i_cell + self.i0_bp * (e_bp - 1.0)
        if not derivs:
            return i_tot, v_sub

        dmult = sp.bishop_a * sp.bishop_m * q_m * q / sp.bishop_vbr_v
        di_cell = -i_0 * e / a_t - mult / rsh - (v_d / rsh) * dmult
        dv_sub = 1.0 - sp.rs_substring_ohm * di_cell
        di_bp = -self.i0_bp * e_bp / A_BYPASS_V * dv_sub
        return i_tot, v_sub, di_cell + di_bp

    def table(self, n_grid: int = N_JUNCTION_GRID):
        """(I, V_sub, v_d) on a junction-voltage grid, one row per condition, current rising along rows."""
        frac = np.linspace(0.0, 1.0, n_grid)
        v_d = -1.4 + (self.v_oc + 1.75)[:, None] * frac[None, :]
        i_tot, v_sub = self.current(v_d, col=True, derivs=False)
        # Current falls as the junction voltage rises; flip so it rises along the row
        i_tot = np.maximum.accumulate(np.clip(i_tot, -50.0, 50.0)[:, ::-1], axis=1)
        return _RowInterp(i_tot), v_sub[:, ::-1], v_d[:, ::-1]

    def voltage_at(self, i_a: np.ndarray, table) -> np.ndarray:
        """Exact sub-string voltage at current i_a (1-D), safeguarded Newton from the table guess."""
        interp, _, vd_tab = table
        h = 2.0 * (vd_tab[:, 0] - vd_tab[:, 1])
        v = interp(i_a[:, None], vd_tab)[:, 0]
        lo, hi = v - h, v + h
        for _ in range(4):
            f, _, df = self.current(v)
            f = f - i_a
            lo = np.where(f > 0.0, v, lo)  # current too high -> junction voltage must rise
            hi = np.where(f > 0.0, hi, v)
            step = np.where(df < 0.0, f / np.where(df < 0.0, df, -1.0), 0.0)
            v_new = v - step
            v = np.where((v_new >= lo) & (v_new <= hi) & (df < 0.0), v_new, 0.5 * (lo + hi))
        return self.current(v)[1]


class _RowInterp:
    """Row-wise linear interpolation over xp (n, m), non-decreasing along each row.

    Rows are normalised to [0, 1] and offset by 2 x row index, so one `searchsorted` over the
    flattened keys serves every row at once.
    """

    def __init__(self, xp: np.ndarray):
        n, self.m = xp.shape
        self.xp = xp
        self.rows = np.arange(n)[:, None]
        self.lo = xp[:, :1]
        self.span = np.maximum(xp[:, -1:] - self.lo, 1e-9)
        self.key = ((xp - self.lo) / self.span + 2.0 * self.rows).ravel()

    def __call__(self, xq: np.ndarray, fp: np.ndarray) -> np.ndarray:
        q = np.clip((xq - self.lo) / self.span, 0.0, 1.0) + 2.0 * self.rows
        idx = np.searchsorted(self.key, q.ravel(), side="right").reshape(xq.shape) - self.m * self.rows
        idx = np.clip(idx, 1, self.m - 1)
        x0 = np.take_along_axis(self.xp, idx - 1, axis=1)
        x1 = np.take_along_axis(self.xp, idx, axis=1)
        f0 = np.take_along_axis(fp, idx - 1, axis=1)
        f1 = np.take_along_axis(fp, idx, axis=1)
        w = np.clip((xq - x0) / np.maximum(x1 - x0, 1e-12), 0.0, 1.0)
        return f0 + w * (f1 - f0)


def _broadcast(n: int, value) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()


class _StringCurves:
    """V_string(I) for a block of operating conditions (one row each).

    Up to three sub-string groups share the string current: healthy, PID-affected (low shunt
    resistance) and shaded (lower irradiance). Shorted sub-strings contribute 0 V. `approx` uses
    interpolated junction-voltage tables (for the coarse global search); `exact` solves every group's
    implicit diode equation at the requested current.
    """

    def __init__(self, g, t, n_short, extra_rs, n_pid_sub, rsh_pid_module, n_shaded, g_shaded, soiling, i_scale, v_scale, spec):
        self.spec = spec
        clean = 1.0 - np.clip(soiling, 0.0, 0.95)
        self.g_unshaded = g * i_scale * clean
        self.t = t
        self.n_healthy = spec.n_substrings - n_short - n_pid_sub - n_shaded
        self.extra_rs = extra_rs
        self.v_scale = v_scale
        rsh_h = np.full_like(g, spec.rsh_substring_healthy_ohm)
        self.groups = [(self.n_healthy, _Substring(self.g_unshaded, t, rsh_h, spec))]
        if np.any(n_pid_sub > 0):
            self.groups.append((n_pid_sub, _Substring(self.g_unshaded, t, rsh_pid_module / spec.substrings_per_module, spec)))
        if np.any(n_shaded > 0):
            self.groups.append((n_shaded, _Substring(g_shaded * i_scale * clean, t, rsh_h, spec)))
        self.tables = [sub.table() for _, sub in self.groups]

    def i_max(self) -> np.ndarray:
        dt = self.t - 25.0
        return self.spec.i_sc_stc_a * (self.g_unshaded / 1000.0) * (1.0 + (self.spec.alpha_isc_pct_per_k / 100.0) * dt) * 1.02

    def approx(self, i_a: np.ndarray) -> np.ndarray:
        v = np.zeros_like(i_a)
        for (n_g, _), (interp, v_tab, _) in zip(self.groups, self.tables):
            v = v + n_g[:, None] * interp(i_a, v_tab)
        return self.v_scale[:, None] * v - i_a * self.extra_rs[:, None]

    def exact(self, i_a: np.ndarray) -> np.ndarray:
        v = np.zeros_like(i_a)
        for (n_g, sub), table in zip(self.groups, self.tables):
            v = v + n_g * sub.voltage_at(i_a, table)
        return self.v_scale * v - i_a * self.extra_rs


def string_mppt(
    g_poa_wm2,
    t_cell_c,
    spec: InverterStringSpec = InverterStringSpec(),
    *,
    n_shorted=0,
    extra_rs_ohm=0.0,
    n_pid_modules=0,
    pid_rsh_module_ohm=400.0,
    n_shaded_substrings=0,
    g_shaded_wm2=None,
    soiling_loss_frac=0.0,
    i_scale=1.0,
    v_scale=1.0,
    n_coarse: int = 120,
    n_golden: int = 18,
    chunk: int = 3000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global maximum power point (V, I, P) of a faulted string, vectorised over operating conditions.

    All fault arguments broadcast against `g_poa_wm2`. `i_scale` and `v_scale` describe string-level
    manufacturing mismatch (photocurrent and voltage scale factors). The global maximum is bracketed
    on a coarse uniform current grid and then located by golden-section search on the exact curve,
    so the result is the global (not a local) maximum and varies smoothly with G and T_c.
    """
    g = np.atleast_1d(np.asarray(g_poa_wm2, dtype=float))
    n = g.size
    t = _broadcast(n, t_cell_c)
    n_short = _broadcast(n, n_shorted)
    rs = np.maximum(_broadcast(n, extra_rs_ohm), 0.0)
    n_pid_sub = _broadcast(n, n_pid_modules) * spec.substrings_per_module
    rsh_pid = _broadcast(n, pid_rsh_module_ohm)
    n_sh = _broadcast(n, n_shaded_substrings)
    g_sh = g.copy() if g_shaded_wm2 is None else _broadcast(n, g_shaded_wm2)
    soil = _broadcast(n, soiling_loss_frac)
    i_sc = _broadcast(n, i_scale)
    v_sc = _broadcast(n, v_scale)

    v_out = np.zeros(n)
    i_out = np.zeros(n)
    lit = g * i_sc * (1.0 - soil) > 1.0
    single = np.flatnonzero(lit & (n_pid_sub == 0) & (n_sh == 0))
    multi = np.flatnonzero(lit & ((n_pid_sub > 0) | (n_sh > 0)))
    frac_c = np.linspace(0.0, 1.0, n_coarse)

    # One sub-string group (healthy, shorted diodes, R_s, soiling): parametrise by junction voltage
    for start in range(0, single.size, chunk):
        rows = single[start : start + chunk]
        clean = 1.0 - np.clip(soil[rows], 0.0, 0.95)
        sub = _Substring(g[rows] * i_sc[rows] * clean, t[rows], np.full(rows.size, spec.rsh_substring_healthy_ohm), spec)
        n_h = (spec.n_substrings - n_short[rows]) * v_sc[rows]
        r_ext = rs[rows]
        interp, v_tab, vd_tab = sub.table(n_grid=160)
        i_tab = interp.xp
        p_tab = np.maximum(n_h[:, None] * v_tab - i_tab * r_ext[:, None], 0.0) * np.maximum(i_tab, 0.0)
        j = np.argmax(p_tab, axis=1)
        idx = np.arange(rows.size)
        a = vd_tab[idx, np.minimum(j + 1, vd_tab.shape[1] - 1)]
        b = vd_tab[idx, np.maximum(j - 1, 0)]

        def operating_point(v_d):
            i_a, v_sub = sub.current(v_d, derivs=False)
            return np.maximum(n_h * v_sub - i_a * r_ext, 0.0), np.maximum(i_a, 0.0)

        def power(v_d):
            v_a, i_a = operating_point(v_d)
            return v_a * i_a

        v_out[rows], i_out[rows] = operating_point(_golden_max(power, a, b, n_golden))

    # Several groups (PID, shading): global search on a current grid, exact solve per group
    for start in range(0, multi.size, chunk):
        rows = multi[start : start + chunk]
        curves = _StringCurves(
            g[rows], t[rows], n_short[rows], rs[rows], n_pid_sub[rows], rsh_pid[rows],
            n_sh[rows], g_sh[rows], soil[rows], i_sc[rows], v_sc[rows], spec,
        )
        i_top = curves.i_max()
        step = i_top / (n_coarse - 1)
        i_c = i_top[:, None] * frac_c[None, :]
        p_c = np.maximum(curves.approx(i_c), 0.0) * i_c
        j = np.argmax(p_c, axis=1)
        a = np.clip(i_c[np.arange(rows.size), j] - step, 0.0, None)
        b = np.minimum(a + 2.0 * step, i_top)
        i_best = _golden_max(lambda i_a: np.maximum(curves.exact(i_a), 0.0) * i_a, a, b, n_golden)
        v_out[rows] = np.maximum(curves.exact(i_best), 0.0)
        i_out[rows] = i_best

    i_channel = i_out * spec.strings_per_mppt
    return v_out, i_channel, v_out * i_channel


def _golden_max(f, a: np.ndarray, b: np.ndarray, n_iter: int) -> np.ndarray:
    """Element-wise golden-section search for the maximum of f on [a, b]."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(n_iter):
        left = f1 >= f2
        b = np.where(left, x2, b)
        a = np.where(left, a, x1)
        x_new = np.where(left, b - inv_phi * (b - a), a + inv_phi * (b - a))
        f_new = f(x_new)
        x2, f2, x1, f1 = (
            np.where(left, x1, x_new), np.where(left, f1, f_new),
            np.where(left, x_new, x2), np.where(left, f_new, f2),
        )
    return 0.5 * (a + b)


def _fault_kwargs(fault: SubstringFaultState, g_poa_wm2: float) -> dict:
    return dict(
        n_shorted=fault.n_shorted_bypass_diodes,
        extra_rs_ohm=fault.extra_rs_ohm,
        n_pid_modules=fault.n_pid_modules,
        pid_rsh_module_ohm=fault.pid_rsh_module_ohm,
        n_shaded_substrings=fault.n_shaded_substrings,
        g_shaded_wm2=g_poa_wm2 * np.clip(fault.shaded_transmission_frac, 0.02, 1.0),
        soiling_loss_frac=fault.soiling_loss_frac,
    )


def sweep_string_iv(
    g_poa_wm2: float,
    t_cell_c: float,
    fault: SubstringFaultState = SubstringFaultState(),
    spec: InverterStringSpec = InverterStringSpec(),
    n_points: int = 400,
) -> dict[str, np.ndarray | float]:
    """Sweep the full I-V and P-V characteristic of a 72-sub-string PV string (or parallel string pair).

    Returns a dictionary containing arrays `v_v`, `i_a`, `p_w`, the global MPPT operating point
    (`v_mppt_v`, `i_mppt_a`, `p_mppt_w`) and the largest secondary local maximum of P(V)
    (`v_local_v`, `p_local_w`; equal to the global point when the curve has a single peak).
    """
    if float(g_poa_wm2) * (1.0 - fault.soiling_loss_frac) <= 1.0:
        zeros = np.zeros(n_points, dtype=float)
        return {
            "v_v": zeros, "i_a": zeros, "p_w": zeros,
            "v_mppt_v": 0.0, "i_mppt_a": 0.0, "p_mppt_w": 0.0,
            "v_local_v": 0.0, "p_local_w": 0.0,
        }

    kw = _fault_kwargs(fault, float(g_poa_wm2))
    v_mp, i_mp, p_mp = string_mppt(g_poa_wm2, t_cell_c, spec, **kw)

    one = lambda x: np.atleast_1d(np.asarray(x, dtype=float))
    curves = _StringCurves(
        one(g_poa_wm2), one(t_cell_c), one(kw["n_shorted"]), one(max(kw["extra_rs_ohm"], 0.0)),
        one(kw["n_pid_modules"] * spec.substrings_per_module), one(kw["pid_rsh_module_ohm"]),
        one(kw["n_shaded_substrings"]), one(kw["g_shaded_wm2"]), one(kw["soiling_loss_frac"]),
        one(1.0), one(1.0), spec,
    )
    i_string = np.linspace(0.0, float(curves.i_max()[0]), n_points)[None, :]
    v_string = curves.approx(i_string)[0]
    valid = v_string >= 0.0
    v_curve = v_string[valid][::-1]
    i_curve = i_string[0][valid][::-1] * spec.strings_per_mppt
    p_curve = v_curve * i_curve

    # Secondary local maxima of P(V), e.g. the unbypassed branch under partial shading
    v_local, p_local = float(v_mp[0]), float(p_mp[0])
    inner = np.flatnonzero((p_curve[1:-1] > p_curve[:-2]) & (p_curve[1:-1] >= p_curve[2:])) + 1
    others = [j for j in inner if abs(v_curve[j] - v_mp[0]) > 0.05 * max(v_mp[0], 1.0)]
    if others:
        j = max(others, key=lambda idx: p_curve[idx])
        if p_curve[j] > 0.15 * p_mp[0]:
            v_local, p_local = float(v_curve[j]), float(p_curve[j])

    return {
        "v_v": v_curve,
        "i_a": i_curve,
        "p_w": p_curve,
        "v_mppt_v": float(v_mp[0]),
        "i_mppt_a": float(i_mp[0]),
        "p_mppt_w": float(p_mp[0]),
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
    v, i, p = string_mppt(g_poa_wm2, t_cell_c, spec, **_fault_kwargs(fault, float(g_poa_wm2)))
    return float(v[0]), float(i[0]), float(p[0])
