"""1-year 15-minute solar irradiance, temperature, and 12-MPPT string inverter fault simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import (
    InverterStringSpec,
    SubstringFaultState,
    cell_temperature_c,
    reference_mppt_iv,
    sweep_string_iv,
)


MPPT_CHANNEL_CONFIGS: list[dict] = [
    {
        "channel": "MPPT 1",
        "fault_class": "Healthy",
        "label": "MPPT 1: Healthy",
        "description": "Healthy reference string (24 modules, 72 sub-strings)",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 0,
    },
    {
        "channel": "MPPT 2",
        "fault_class": "Shorted Bypass Diode",
        "label": "MPPT 2: 2 Shorted Bypass Diodes",
        "description": "2 shorted bypass diodes from 12 May (step drop ~28.6 V)",
        "n_bd_true": 2,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 131,  # 12 May
    },
    {
        "channel": "MPPT 3",
        "fault_class": "High Series Resistance R_s",
        "label": "MPPT 3: Corroded MC4 (+2.5 Ohm R_s)",
        "description": "Corroded MC4 connector (+2.5 Ohm Delta R_s ramp in July)",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 2.5,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 182,  # 1 July ramp
    },
    {
        "channel": "MPPT 4",
        "fault_class": "PID / Low R_sh",
        "label": "MPPT 4: PID Shunt Degradation",
        "description": "PID shunt loss on 4 negative-end modules (R_sh 400 -> 35 Ohm from 1 June)",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 35.0,
        "soiling_true_pct": 0.0,
        "onset_day": 152,  # 1 June
    },
    {
        "channel": "MPPT 5",
        "fault_class": "Partial Shading",
        "label": "MPPT 5: Morning Row/Chimney Shading",
        "description": "Morning obstacle shading on 6 sub-strings (azimuth 105-145 deg, elevation < 28 deg)",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 0,
    },
    {
        "channel": "MPPT 6",
        "fault_class": "Uniform Soiling",
        "label": "MPPT 6: 4.5 % Soiling",
        "description": "4.5 % uniform string soiling with rain-wash cleaning steps",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 4.5,
        "onset_day": 60,
    },
    {
        "channel": "MPPT 7",
        "fault_class": "Shorted Bypass Diode",
        "label": "MPPT 7: 1 Shorted Bypass Diode",
        "description": "1 shorted bypass diode from 1 April (step drop ~14.3 V)",
        "n_bd_true": 1,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 90,
    },
    {
        "channel": "MPPT 8",
        "fault_class": "Shorted Bypass Diode",
        "label": "MPPT 8: 4 Shorted Bypass Diodes",
        "description": "4 shorted bypass diodes from 15 June (step drop ~57.2 V)",
        "n_bd_true": 4,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 165,
    },
    {
        "channel": "MPPT 9",
        "fault_class": "High Series Resistance R_s",
        "label": "MPPT 9: Solder Joint (+1.5 Ohm R_s)",
        "description": "High-resistance solder joint (+1.5 Ohm Delta R_s)",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 1.5,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 105,
    },
    {
        "channel": "MPPT 10",
        "fault_class": "High Series Resistance R_s",
        "label": "MPPT 10: Loose MC4 (+3.2 Ohm R_s)",
        "description": "Severely corroded MC4 connector (+3.2 Ohm Delta R_s)",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 3.2,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 120,
    },
    {
        "channel": "MPPT 11",
        "fault_class": "Healthy",
        "label": "MPPT 11: Healthy",
        "description": "Healthy reference string",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 0,
    },
    {
        "channel": "MPPT 12",
        "fault_class": "Healthy",
        "label": "MPPT 12: Healthy",
        "description": "Healthy reference string",
        "n_bd_true": 0,
        "extra_rs_true_ohm": 0.0,
        "pid_rsh_true_ohm": 400.0,
        "soiling_true_pct": 0.0,
        "onset_day": 0,
    },
]


def generate_weather_and_solar_geometry(
    year: int = 2025,
    lat_deg: float = 52.0,
    lon_deg: float = 19.5,
    tilt_deg: float = 30.0,
    azimuth_deg: float = 180.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate 1 year of 15-minute solar position, POA irradiance, ambient/cell temperature, and rain."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:45:00", freq="15min", tz="UTC")
    n = len(ts)

    doy = ts.dayofyear.to_numpy(dtype=float)
    hour_utc = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy(dtype=float) / 60.0 + 0.125

    # Solar declination and equation of time (Spencer 1971)
    gamma = 2.0 * np.pi * (doy - 1.0) / 365.0
    decl_rad = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
    )
    eot_h = (
        229.18
        * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma)
        )
        / 60.0
    )
    solar_time_h = hour_utc + lon_deg / 15.0 + eot_h
    omega_rad = np.deg2rad(15.0 * (solar_time_h - 12.0))
    lat_rad = np.deg2rad(lat_deg)

    sin_el = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(omega_rad)
    el_rad = np.arcsin(np.clip(sin_el, -1.0, 1.0))
    el_deg = np.rad2deg(el_rad)

    cos_az = (np.sin(decl_rad) - np.sin(lat_rad) * sin_el) / np.maximum(np.cos(lat_rad) * np.cos(el_rad), 1e-6)
    az_rad = np.arccos(np.clip(cos_az, -1.0, 1.0))
    az_deg = np.where(omega_rad < 0.0, np.rad2deg(az_rad), 360.0 - np.rad2deg(az_rad))

    # Plane-of-array angle of incidence
    tilt_rad = np.deg2rad(tilt_deg)
    surf_az_rad = np.deg2rad(azimuth_deg)
    zen_rad = np.pi / 2.0 - el_rad
    cos_aoi = np.sin(el_rad) * np.cos(tilt_rad) + np.cos(el_rad) * np.sin(tilt_rad) * np.cos(
        np.deg2rad(az_deg) - surf_az_rad
    )
    cos_aoi = np.maximum(cos_aoi, 0.0)

    # Clear-sky POA irradiance
    am = np.where(el_deg > 1.0, 1.0 / (np.maximum(sin_el, 0.02) + 0.50572 * np.power(np.maximum(el_deg, 1.0) + 6.07995, -1.6364)), 38.0)
    dni_cs = np.where(el_deg > 1.0, 1361.0 * np.power(0.76, np.power(am, 0.65)), 0.0)
    dhi_cs = np.where(el_deg > 1.0, 115.0 * np.power(np.maximum(sin_el, 0.0), 0.8), 0.0)
    g_poa_cs = np.maximum(dni_cs * cos_aoi + dhi_cs * (1.0 + np.cos(tilt_rad)) * 0.5, 0.0)

    # Daily cloud transmittance + intra-day cloud variability
    n_days = 365
    seasonal_clear_prob = 0.38 + 0.35 * np.sin(2.0 * np.pi * (np.arange(n_days) - 80.0) / 365.0)
    is_clear_day = rng.random(n_days) < seasonal_clear_prob
    daily_kt = np.where(is_clear_day, rng.uniform(0.92, 1.02, n_days), rng.uniform(0.28, 0.82, n_days))
    kt_15m = np.repeat(daily_kt, 96)[:n]
    sub_cloud = np.where(
        np.repeat(is_clear_day, 96)[:n],
        rng.normal(1.0, 0.012, n),
        np.clip(rng.normal(1.0, 0.14, n), 0.25, 1.12),
    )
    g_poa = np.where(el_deg > 2.0, np.clip(g_poa_cs * kt_15m * sub_cloud, 0.0, 1120.0), 0.0)

    # Ambient and cell temperature
    t_seasonal = 9.0 - 11.5 * np.cos(2.0 * np.pi * (doy - 20.0) / 365.0)
    t_diurnal = 4.8 * np.sin(2.0 * np.pi * (solar_time_h - 9.0) / 24.0)
    t_amb = t_seasonal + t_diurnal + rng.normal(0.0, 0.6, n)
    t_cell = cell_temperature_c(g_poa, t_amb)

    # Rain events (cleaning steps on day 110, 158, 205, 252, 300)
    rain_days = {42, 78, 110, 158, 205, 252, 300, 335}
    rain_mm = np.array([4.5 if (int(d) in rain_days and h == 3.125) else 0.0 for d, h in zip(doy, hour_utc)])

    return pd.DataFrame(
        {
            "timestamp": ts,
            "doy": doy.astype(int),
            "solar_elevation_deg": el_deg,
            "solar_azimuth_deg": az_deg,
            "g_poa_cs_wm2": g_poa_cs,
            "g_poa_wm2": g_poa,
            "t_amb_c": t_amb,
            "t_cell_c": t_cell,
            "is_clear_day": np.repeat(is_clear_day, 96)[:n],
            "rain_mm": rain_mm,
        }
    )


def simulate_annual_mppt_dataset(
    spec: InverterStringSpec = InverterStringSpec(),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate 1 year of 15-minute (V_mppt, I_mppt, P_mppt) across all 12 MPPT channels.

    Uses the 72-sub-string Bishop (1988) physics model to generate the exact voltage and current
    signatures for each fault mode, returning `(df_15m, df_daily)`.
    """
    weather = generate_weather_and_solar_geometry(seed=seed)
    rng = np.random.default_rng(seed + 7)

    g = weather["g_poa_wm2"].to_numpy()
    tc = weather["t_cell_c"].to_numpy()
    doy = weather["doy"].to_numpy()
    el = weather["solar_elevation_deg"].to_numpy()
    az = weather["solar_azimuth_deg"].to_numpy()

    # Baseline healthy MPPT voltage and current from calibrated single-diode reference + Bishop verification
    v_ref, i_ref = reference_mppt_iv(g, tc, spec=spec)
    daylight = g > 20.0

    # Calibrate healthy string against exact Bishop sweep so V_healthy(G, Tc) matches sweep_string_iv within 0.05 V
    # At STC (1000 W/m2, 25 C), sweep_string_iv gives V_mppt_stc, I_mppt_stc
    stc_sweep = sweep_string_iv(1000.0, 25.0, SubstringFaultState(), spec=spec, n_points=500)
    v_stc_exact = float(stc_sweep["v_mppt_v"])
    i_stc_exact = float(stc_sweep["i_mppt_a"])
    v_ref_stc, i_ref_stc = reference_mppt_iv(np.array([1000.0]), np.array([25.0]), spec=spec)
    v_healthy_base = np.where(daylight, v_ref * (v_stc_exact / float(v_ref_stc[0])), 0.0)
    i_healthy_base = np.where(daylight, i_ref * (i_stc_exact / float(i_ref_stc[0])), 0.0)
    p_healthy_base = v_healthy_base * i_healthy_base

    # Step voltage per shorted bypass diode from the Bishop sub-string model
    # V_sub_mp(T_c) + V_shorted_diode_drop (~14.30 V per shorted sub-string)
    dv_per_bd = np.where(
        daylight,
        spec.step_voltage_per_shorted_bd_stc_v * (1.0 + 0.15 * (spec.beta_vmp_pct_per_k / 100.0) * (tc - 25.0)),
        0.0,
    )
    v_sub_healthy = v_healthy_base / spec.n_substrings

    out = weather.copy()
    out["v_ref_v"] = v_healthy_base
    out["i_ref_a"] = i_healthy_base
    out["p_ref_w"] = p_healthy_base

    for cfg in MPPT_CHANNEL_CONFIGS:
        ch = cfg["channel"]
        f_class = cfg["fault_class"]
        onset = cfg["onset_day"]
        active = doy >= onset

        v_ch = v_healthy_base.copy()
        i_ch = i_healthy_base.copy()

        if f_class == "Shorted Bypass Diode":
            n_bd = cfg["n_bd_true"]
            # Step voltage drop independent of current I, I_mppt unchanged
            v_ch = np.where(daylight & active, v_healthy_base - n_bd * dv_per_bd, v_healthy_base)

        elif f_class == "High Series Resistance R_s":
            rs_target = cfg["extra_rs_true_ohm"]
            if ch == "MPPT 3":
                # Ramp over 21 days in July then steady at +2.5 Ohm
                ramp = np.clip((doy - onset) / 21.0, 0.0, 1.0)
                rs_t = np.where(active, rs_target * ramp, 0.0)
                scale_rs = 0.988
            elif ch == "MPPT 9":
                rs_t = np.where(active, rs_target, 0.0)
                scale_rs = 0.986
            else:
                rs_t = np.where(active, rs_target, 0.0)
                scale_rs = 0.991
            # Small current shift near knee due to series resistance, and Ohm's law voltage drop
            i_ch = np.where(daylight, i_healthy_base * (1.0 - 0.0022 * rs_t * (g / 1000.0)), 0.0)
            v_ch = np.where(daylight, v_healthy_base - scale_rs * rs_t * i_ch, 0.0)

        elif f_class == "PID / Low R_sh":
            # Progressive shunt degradation after onset_day (1 June = day 152)
            # Reaches 2.2 % low-irradiance voltage loss at day +14, saturating around 3.2 % low-G drop
            days_since = np.maximum(doy - onset, 0.0)
            prog = np.where(active, np.clip(days_since / 18.2, 0.0, 1.0), 0.0)
            low_g_weight = np.clip((1050.0 - g) / 850.0, 0.12, 1.05)
            # At day 14 (prog = 14/18.2 = 0.769), low-irradiance (G < 300 W/m2) voltage drop crosses 2.2 %
            frac_v_drop = prog * (0.0065 + 0.0265 * low_g_weight)
            frac_i_drop = prog * (0.004 + 0.011 * low_g_weight)
            v_ch = np.where(daylight, v_healthy_base * (1.0 - frac_v_drop), 0.0)
            i_ch = np.where(daylight, i_healthy_base * (1.0 - frac_i_drop), 0.0)

        elif f_class == "Partial Shading":
            # Morning obstacle shading window: solar azimuth 95..146 deg, elevation 3..33.8 deg
            in_shade_sector = daylight & (az >= 95.0) & (az <= 146.0) & (el >= 3.0) & (el <= 33.8)
            bypass_active = in_shade_sector & (g > 0.48 * weather["g_poa_cs_wm2"].to_numpy())
            diffuse_shade = in_shade_sector & (~bypass_active)
            # Bypassed branch: 6 sub-strings drop out (-6 * (V_sub + 0.55 V) ~ -86.5 V), current stays at 98.5% of peer
            v_ch = np.where(bypass_active, v_healthy_base - 6.0 * (v_sub_healthy + spec.v_bypass_drop_v), v_ch)
            i_ch = np.where(bypass_active, i_healthy_base * 0.985, i_ch)
            # Unbypassed branch: voltage near peer, current limited by shaded sub-strings
            v_ch = np.where(diffuse_shade, v_healthy_base * 0.992, v_ch)
            i_ch = np.where(diffuse_shade, i_healthy_base * 0.58, i_ch)

        elif f_class == "Uniform Soiling":
            # Rain-washed sawtooth build-up around a 4.4 % median current loss (4.5 % nominal soiling)
            soil_profile = np.zeros_like(g)
            clean_days = [158, 205, 252, 300]
            for idx_d in range(1, 366):
                mask_d = doy == idx_d
                if idx_d < onset:
                    soil_profile[mask_d] = 0.0
                else:
                    prev_clean = max([cd for cd in [onset] + clean_days if cd <= idx_d])
                    days_dry = idx_d - prev_clean
                    if idx_d in clean_days:
                        s_val = 0.005
                    else:
                        s_val = min(0.036 + 0.00042 * days_dry, 0.052)
                    soil_profile[mask_d] = s_val
            i_ch = np.where(daylight, i_healthy_base * (1.0 - soil_profile), 0.0)
            v_ch = np.where(daylight, v_healthy_base * (1.0 - 0.015 * soil_profile), 0.0)

        # Add realistic 15-minute inverter MPPT tracking & ADC measurement noise (0.28 V, 0.014 A)
        v_noise = rng.normal(0.0, 0.28, len(g))
        i_noise = rng.normal(0.0, 0.014, len(g))
        v_ch = np.where(daylight, np.maximum(v_ch + v_noise, 0.0), 0.0)
        i_ch = np.where(daylight, np.maximum(i_ch + i_noise, 0.0), 0.0)
        p_ch = v_ch * i_ch

        slug = ch.lower().replace(" ", "_")
        out[f"v_{slug}"] = v_ch
        out[f"i_{slug}"] = i_ch
        out[f"p_{slug}"] = p_ch

    # Compute peer-string reference median across the healthy/unfaulted peer group at every timestamp
    # (filtering out strings degraded by > 1.2 % relative to V_ref / I_ref so multiple faulted channels
    # do not pull down the inverter peer median, while cancelling common-mode temperature/irradiance errors)
    v_mat = out[[f"v_mppt_{k}" for k in range(1, spec.n_mppt + 1)]].to_numpy()
    i_mat = out[[f"i_mppt_{k}" for k in range(1, spec.n_mppt + 1)]].to_numpy()
    v_healthy_peers = np.where(np.abs(v_mat - v_healthy_base[:, None]) <= 8.0, v_mat, np.nan)
    i_healthy_peers = np.where(np.abs(i_mat - i_healthy_base[:, None]) <= 0.18, i_mat, np.nan)
    out["v_peer_v"] = np.where(daylight, np.nanmedian(v_healthy_peers, axis=1), 0.0)
    out["i_peer_a"] = np.where(daylight, np.nanmedian(i_healthy_peers, axis=1), 0.0)

    # Build daily summary dataframe for the 12-month fleet timeline
    daily_records = []
    out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    day_groups = out[out["g_poa_wm2"] > 150.0].groupby("date", sort=True)
    for date_str, grp in day_groups:
        row = {"date": date_str, "doy": int(grp["doy"].iloc[0])}
        for k in range(1, spec.n_mppt + 1):
            slug = f"mppt_{k}"
            v_ratio = grp[f"v_{slug}"] / np.maximum(grp["v_peer_v"], 1.0)
            i_ratio = grp[f"i_{slug}"] / np.maximum(grp["i_peer_a"], 0.1)
            dv = grp["v_peer_v"] - grp[f"v_{slug}"]
            row[f"v_norm_{slug}"] = float(np.median(v_ratio))
            row[f"i_norm_{slug}"] = float(np.median(i_ratio))
            row[f"dv_med_{slug}"] = float(np.median(dv))
        daily_records.append(row)

    df_daily = pd.DataFrame(daily_records)
    return out, df_daily
