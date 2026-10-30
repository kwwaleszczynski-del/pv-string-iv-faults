"""Synthetic 15-minute weather and a 12-MPPT string inverter with physically injected string faults.

This module is the ground-truth generator. Every telemetry value it emits comes from
`model.string_mppt`: each fault is a change of a physical parameter of the sub-string model
(shorted sub-strings, extra series resistance, low shunt resistance on PID-affected modules,
diffuse-only irradiance on obstacle-shaded sub-strings, soiling transmission loss). Strings differ
by manufacturing mismatch and cable resistance, and the telemetry carries sensor noise. The
estimator in `diagnostics.py` never sees anything returned here except `SimulationResult.telemetry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model import InverterStringSpec, cell_temperature_c, string_mppt
from .solar import clear_sky_poa, solar_position

FAULT_FAMILIES: list[str] = [
    "Shorted Bypass Diode",
    "High Series Resistance R_s",
    "PID / Low R_sh",
    "Partial Shading",
    "Uniform Soiling",
]
RAIN_CLEANING_MM = 2.0
SHADED_SKY_VIEW = 0.9  # fraction of the diffuse irradiance still reaching sub-strings behind the obstacle


@dataclass(frozen=True)
class Site:
    lat_deg: float = 52.0
    lon_deg: float = 19.5
    tilt_deg: float = 30.0
    azimuth_deg: float = 180.0


@dataclass(frozen=True)
class ChannelScenario:
    """Physical fault scenario of one MPPT channel (one string). Days are day-of-year (1..365)."""

    channel: str
    fault_class: str = "Healthy"
    label: str = ""
    onset_day: int = 1
    n_bd: int = 0
    rs_final_ohm: float = 0.0
    rs_ramp_days: float = 0.0
    n_pid_modules: int = 0
    pid_rsh_final_ohm: float = 400.0
    pid_tau_days: float = 20.0
    n_shaded_substrings: int = 0
    shade_az_deg: tuple[float, float] = (0.0, 0.0)
    shade_top_el_deg: float = 0.0
    soiling_rate_pct_per_day: float = 0.0
    soiling_max_pct: float = 0.0
    rain_clean_frac: float = 0.0


SHOWCASE_SCENARIOS: list[ChannelScenario] = [
    ChannelScenario("MPPT 1", label="MPPT 1: Healthy"),
    ChannelScenario("MPPT 2", "Shorted Bypass Diode", "MPPT 2: 2 shorted bypass diodes (12 May)", onset_day=132, n_bd=2),
    ChannelScenario(
        "MPPT 3", "High Series Resistance R_s", "MPPT 3: Corroded MC4, +2.5 Ohm over 21 days (1 Jul)",
        onset_day=182, rs_final_ohm=2.5, rs_ramp_days=21.0,
    ),
    ChannelScenario(
        "MPPT 4", "PID / Low R_sh", "MPPT 4: PID on 4 modules, R_sh 400 -> 35 Ohm (1 Jun)",
        onset_day=152, n_pid_modules=4, pid_rsh_final_ohm=35.0, pid_tau_days=20.0,
    ),
    ChannelScenario(
        "MPPT 5", "Partial Shading", "MPPT 5: Morning obstacle shading 6 sub-strings",
        n_shaded_substrings=6, shade_az_deg=(100.0, 150.0), shade_top_el_deg=25.0,
    ),
    ChannelScenario(
        "MPPT 6", "Uniform Soiling", "MPPT 6: Soiling 0.08 %/day, rain-washed (1 Mar)",
        onset_day=60, soiling_rate_pct_per_day=0.08, soiling_max_pct=6.0, rain_clean_frac=0.6,
    ),
    ChannelScenario("MPPT 7", "Shorted Bypass Diode", "MPPT 7: 1 shorted bypass diode (1 Apr)", onset_day=91, n_bd=1),
    ChannelScenario("MPPT 8", "Shorted Bypass Diode", "MPPT 8: 4 shorted bypass diodes (15 Jun)", onset_day=166, n_bd=4),
    ChannelScenario(
        "MPPT 9", "High Series Resistance R_s", "MPPT 9: Solder joint +1.5 Ohm (15 Apr)", onset_day=105, rs_final_ohm=1.5,
    ),
    ChannelScenario(
        "MPPT 10", "High Series Resistance R_s", "MPPT 10: Corroded MC4 +3.2 Ohm (30 Apr)", onset_day=120, rs_final_ohm=3.2,
    ),
    ChannelScenario("MPPT 11", label="MPPT 11: Healthy"),
    ChannelScenario("MPPT 12", label="MPPT 12: Healthy"),
]


@dataclass
class SimulationResult:
    """`telemetry` is what a SCADA system would store; everything else is ground truth."""

    telemetry: pd.DataFrame
    truth: pd.DataFrame
    scenarios: list[ChannelScenario]
    site: Site
    seed: int
    daily_rain_mm: np.ndarray = field(repr=False)


def generate_weather(site: Site = Site(), year: int = 2025, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic 15-minute weather: clear-sky geometry, random cloud days, beam/diffuse split, rain."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(f"{year}-01-01", periods=365 * 96, freq="15min", tz="UTC")
    n = len(ts)
    doy = ts.day_of_year.to_numpy()
    el, az = solar_position(ts, site.lat_deg, site.lon_deg)
    beam_cs, diffuse_cs = clear_sky_poa(el, az, site.tilt_deg, site.azimuth_deg)
    g_cs = beam_cs + diffuse_cs

    days = np.arange(365)
    seasonal_clear_prob = 0.38 + 0.35 * np.sin(2.0 * np.pi * (days - 80.0) / 365.0)
    is_clear_day = rng.random(365) < seasonal_clear_prob
    daily_kt = np.where(is_clear_day, rng.uniform(0.92, 1.02, 365), rng.uniform(0.28, 0.82, 365))
    clear_15m = np.repeat(is_clear_day, 96)
    sub_cloud = np.where(clear_15m, rng.normal(1.0, 0.012, n), np.clip(rng.normal(1.0, 0.14, n), 0.25, 1.12))
    k_t = np.repeat(daily_kt, 96) * sub_cloud

    g_poa = np.where(el > 2.0, np.clip(g_cs * k_t, 0.0, 1120.0), 0.0)
    # Clouds remove the beam faster than the total: beam transmittance reaches zero at k_t = 0.3
    beam = np.minimum(beam_cs * np.clip((k_t - 0.3) / 0.7, 0.0, 1.05), g_poa)
    diffuse = g_poa - beam

    t_seasonal = 9.0 - 11.5 * np.cos(2.0 * np.pi * (doy - 20.0) / 365.0)
    hour_utc = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    t_diurnal = 4.8 * np.sin(2.0 * np.pi * (hour_utc + site.lon_deg / 15.0 - 9.0) / 24.0)
    t_amb = t_seasonal + t_diurnal + rng.normal(0.0, 0.6, n)

    wet_prob = 0.32 + 0.08 * np.sin(2.0 * np.pi * (days - 120.0) / 365.0)
    daily_rain = np.where(rng.random(365) < wet_prob, rng.exponential(4.0, 365), 0.0)

    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "doy": doy,
            "solar_elevation_deg": el,
            "solar_azimuth_deg": az,
            "g_poa_cs_wm2": g_cs,
            "g_poa_wm2": g_poa,
            "g_beam_poa_wm2": beam,
            "g_diffuse_poa_wm2": diffuse,
            "t_amb_c": t_amb,
            "t_cell_c": cell_temperature_c(g_poa, t_amb),
        }
    )
    return weather, daily_rain


def channel_fault_state(sc: ChannelScenario, weather: pd.DataFrame, daily_rain: np.ndarray, spec: InverterStringSpec) -> dict[str, np.ndarray]:
    """Physical fault parameters of one channel at every 15-minute step."""
    doy = weather["doy"].to_numpy()
    days_since = (doy - sc.onset_day).astype(float)
    active = days_since >= 0.0

    n_bd = np.where(active, sc.n_bd, 0)
    ramp = np.ones_like(days_since) if sc.rs_ramp_days <= 0 else np.clip((days_since + 1.0) / sc.rs_ramp_days, 0.0, 1.0)
    rs = np.where(active, sc.rs_final_ohm * ramp, 0.0)
    healthy_rsh = spec.rsh_module_healthy_ohm
    rsh = np.where(
        active,
        sc.pid_rsh_final_ohm + (healthy_rsh - sc.pid_rsh_final_ohm) * np.exp(-np.maximum(days_since, 0.0) / sc.pid_tau_days),
        healthy_rsh,
    )
    n_pid = np.where(active & (sc.n_pid_modules > 0), sc.n_pid_modules, 0)

    el = weather["solar_elevation_deg"].to_numpy()
    az = weather["solar_azimuth_deg"].to_numpy()
    behind_obstacle = (az >= sc.shade_az_deg[0]) & (az <= sc.shade_az_deg[1]) & (el > 0.0) & (el < sc.shade_top_el_deg)
    n_shaded = np.where(active & behind_obstacle, sc.n_shaded_substrings, 0)

    # Soiling level at the start of each day: linear build-up, partial wash-off on rainy days, capped
    soil_day = np.zeros(365)
    level = 0.0
    for d in range(365):
        soil_day[d] = level
        if d + 1 >= sc.onset_day:
            level = min(level + sc.soiling_rate_pct_per_day / 100.0, sc.soiling_max_pct / 100.0)
            if daily_rain[d] >= RAIN_CLEANING_MM:
                level *= 1.0 - sc.rain_clean_frac
    soiling = soil_day[doy - 1]

    return {"n_bd": n_bd, "rs": rs, "rsh": rsh, "n_pid": n_pid, "n_shaded": n_shaded, "soiling": soiling}


def simulate_inverter(
    scenarios: list[ChannelScenario] = SHOWCASE_SCENARIOS,
    site: Site = Site(),
    seed: int = 42,
    spec: InverterStringSpec = InverterStringSpec(),
    year: int = 2025,
) -> SimulationResult:
    """Simulate one year of 15-minute MPPT telemetry for one inverter and keep the ground truth apart."""
    weather, daily_rain = generate_weather(site, year=year, seed=seed)
    rng = np.random.default_rng(seed + 7)

    g = weather["g_poa_wm2"].to_numpy()
    tc = weather["t_cell_c"].to_numpy()
    g_shaded = SHADED_SKY_VIEW * weather["g_diffuse_poa_wm2"].to_numpy()
    lit = g > 1.0
    n = len(weather)

    # Pyranometer: per-inverter calibration error plus sample noise; thermometer noise
    g_cal = rng.normal(1.0, 0.015)
    telemetry = pd.DataFrame(
        {
            "timestamp": weather["timestamp"],
            "g_poa_wm2": np.where(lit, np.maximum(g * g_cal * (1.0 + rng.normal(0.0, 0.01, n)), 0.0), 0.0),
            "t_amb_c": weather["t_amb_c"].to_numpy() + rng.normal(0.0, 0.3, n),
        }
    )
    truth = weather.copy()

    for k, sc in enumerate(scenarios, start=1):
        # String-level manufacturing mismatch and cable resistance (2 x 40-100 m of 6 mm2 copper)
        i_scale = rng.normal(1.0, 0.004)
        v_scale = rng.normal(1.0, 0.003)
        r_cable = rng.uniform(0.2, 0.6)
        state = channel_fault_state(sc, weather, daily_rain, spec)

        v = np.zeros(n)
        i = np.zeros(n)
        v[lit], i[lit], _ = string_mppt(
            g[lit], tc[lit], spec,
            n_shorted=state["n_bd"][lit],
            extra_rs_ohm=r_cable + state["rs"][lit],
            n_pid_modules=state["n_pid"][lit],
            pid_rsh_module_ohm=state["rsh"][lit],
            n_shaded_substrings=state["n_shaded"][lit],
            g_shaded_wm2=g_shaded[lit],
            soiling_loss_frac=state["soiling"][lit],
            i_scale=i_scale,
            v_scale=v_scale,
        )
        p_cf = v * i
        if sc.fault_class != "Healthy":
            v_cf = np.zeros(n)
            i_cf = np.zeros(n)
            v_cf[lit], i_cf[lit], _ = string_mppt(g[lit], tc[lit], spec, extra_rs_ohm=r_cable, i_scale=i_scale, v_scale=v_scale)
            p_cf = v_cf * i_cf

        # Inverter MPPT tracking and ADC noise on the 15-minute operating point
        v_meas = np.where(lit, np.maximum(v + rng.normal(0.0, 0.8, n), 0.0), 0.0)
        i_meas = np.where(lit, np.maximum(i * (1.0 + rng.normal(0.0, 0.003, n)) + rng.normal(0.0, 0.015, n), 0.0), 0.0)
        telemetry[f"v_mppt_{k}"] = v_meas
        telemetry[f"i_mppt_{k}"] = i_meas

        truth[f"p_true_mppt_{k}"] = v * i
        truth[f"p_counterfactual_mppt_{k}"] = p_cf
        truth[f"soiling_mppt_{k}"] = state["soiling"]
        truth[f"rs_fault_mppt_{k}"] = state["rs"]
        truth[f"rsh_pid_mppt_{k}"] = state["rsh"]
        truth[f"shaded_mppt_{k}"] = state["n_shaded"] > 0

    return SimulationResult(telemetry, truth, list(scenarios), site, seed, daily_rain)


def random_site(rng: np.random.Generator) -> Site:
    return Site(
        lat_deg=float(rng.uniform(49.5, 54.5)),
        lon_deg=float(rng.uniform(14.5, 23.5)),
        tilt_deg=float(rng.uniform(20.0, 35.0)),
        azimuth_deg=float(rng.uniform(165.0, 195.0)),
    )


def random_scenarios(rng: np.random.Generator, n_mppt: int = 12, p_healthy: float = 0.4) -> list[ChannelScenario]:
    """Draw one inverter's channel scenarios: each channel is healthy or carries one random fault."""
    out = []
    for k in range(1, n_mppt + 1):
        ch = f"MPPT {k}"
        if rng.random() < p_healthy:
            out.append(ChannelScenario(ch, label=f"{ch}: Healthy"))
            continue
        family = FAULT_FAMILIES[int(rng.integers(len(FAULT_FAMILIES)))]
        if family == "Shorted Bypass Diode":
            n_bd = int(rng.integers(1, 5))
            onset = int(rng.integers(30, 301))
            out.append(ChannelScenario(ch, family, f"{ch}: {n_bd} shorted bypass diode(s), day {onset}", onset_day=onset, n_bd=n_bd))
        elif family == "High Series Resistance R_s":
            rs = float(rng.uniform(0.8, 4.0))
            ramp = 0.0 if rng.random() < 0.5 else float(rng.uniform(7.0, 45.0))
            onset = int(rng.integers(30, 301))
            out.append(
                ChannelScenario(ch, family, f"{ch}: +{rs:.2f} Ohm R_s, day {onset}", onset_day=onset, rs_final_ohm=rs, rs_ramp_days=ramp)
            )
        elif family == "PID / Low R_sh":
            n_pid = int(rng.integers(2, 7))
            rsh = float(rng.uniform(20.0, 90.0))
            onset = int(rng.integers(30, 251))
            out.append(
                ChannelScenario(
                    ch, family, f"{ch}: PID on {n_pid} modules, R_sh -> {rsh:.0f} Ohm, day {onset}",
                    onset_day=onset, n_pid_modules=n_pid, pid_rsh_final_ohm=rsh, pid_tau_days=float(rng.uniform(10.0, 60.0)),
                )
            )
        elif family == "Partial Shading":
            start = float(rng.uniform(90.0, 125.0)) if rng.random() < 0.5 else float(rng.uniform(215.0, 250.0))
            width = float(rng.uniform(25.0, 50.0))
            n_sh = int(rng.integers(3, 10))
            out.append(
                ChannelScenario(
                    ch, family, f"{ch}: obstacle shading {n_sh} sub-strings, az {start:.0f}-{start + width:.0f} deg",
                    n_shaded_substrings=n_sh, shade_az_deg=(start, start + width), shade_top_el_deg=float(rng.uniform(12.0, 30.0)),
                )
            )
        else:
            rate = float(rng.uniform(0.04, 0.2))
            onset = int(rng.integers(30, 251))
            out.append(
                ChannelScenario(
                    ch, family, f"{ch}: soiling {rate:.2f} %/day, day {onset}",
                    onset_day=onset, soiling_rate_pct_per_day=rate, soiling_max_pct=float(rng.uniform(3.0, 10.0)),
                    rain_clean_frac=float(rng.uniform(0.3, 0.9)),
                )
            )
    return out


def simulate_random_inverter(seed: int, spec: InverterStringSpec = InverterStringSpec()) -> SimulationResult:
    """One synthetic inverter whose site, weather, fault scenarios and string mismatch all derive from `seed`."""
    rng = np.random.default_rng(seed + 100_003)
    site = random_site(rng)
    scenarios = random_scenarios(rng, n_mppt=spec.n_mppt)
    return simulate_inverter(scenarios, site=site, seed=seed, spec=spec)
