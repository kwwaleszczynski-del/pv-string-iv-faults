"""Solar position and clear-sky plane-of-array irradiance (Spencer 1971 declination and equation of time)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def solar_position(timestamps: pd.DatetimeIndex, lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (elevation, azimuth) in degrees at the middle of each 15-minute interval (UTC timestamps).

    Azimuth is measured clockwise from north (90 = east, 180 = south).
    """
    ts = pd.DatetimeIndex(timestamps)
    doy = ts.day_of_year.to_numpy(dtype=float)
    hour_utc = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy(dtype=float) / 60.0 + 0.125

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
    cos_az = (np.sin(decl_rad) - np.sin(lat_rad) * sin_el) / np.maximum(np.cos(lat_rad) * np.cos(el_rad), 1e-6)
    az_rad = np.arccos(np.clip(cos_az, -1.0, 1.0))
    az_deg = np.where(omega_rad < 0.0, np.rad2deg(az_rad), 360.0 - np.rad2deg(az_rad))
    return np.rad2deg(el_rad), az_deg


def clear_sky_poa(
    el_deg: np.ndarray, az_deg: np.ndarray, tilt_deg: float, surface_azimuth_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clear-sky (beam, diffuse) irradiance on a tilted plane in W/m2 (simple air-mass transmittance model)."""
    el_rad = np.deg2rad(el_deg)
    sin_el = np.sin(el_rad)
    tilt_rad = np.deg2rad(tilt_deg)
    cos_aoi = sin_el * np.cos(tilt_rad) + np.cos(el_rad) * np.sin(tilt_rad) * np.cos(
        np.deg2rad(az_deg - surface_azimuth_deg)
    )
    cos_aoi = np.maximum(cos_aoi, 0.0)
    up = el_deg > 1.0
    am = np.where(up, 1.0 / (np.maximum(sin_el, 0.02) + 0.50572 * np.power(np.maximum(el_deg, 1.0) + 6.07995, -1.6364)), 38.0)
    dni = np.where(up, 1361.0 * np.power(0.76, np.power(am, 0.65)), 0.0)
    dhi = np.where(up, 115.0 * np.power(np.maximum(sin_el, 0.0), 0.8), 0.0)
    return dni * cos_aoi, dhi * (1.0 + np.cos(tilt_rad)) * 0.5
