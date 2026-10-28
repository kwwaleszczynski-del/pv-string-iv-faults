"""End-to-end pipeline: simulation, fault diagnosis, verification, Matplotlib figures, and dashboard data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostics import FAULT_CLASSES, diagnose_mppt_channels, evaluate_fault_verification
from .model import InverterStringSpec, SubstringFaultState, sweep_string_iv
from .simulator import MPPT_CHANNEL_CONFIGS, simulate_annual_mppt_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.8)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def generate_plots(
    df_15m: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_diag: pd.DataFrame,
    verification: dict,
    spec: InverterStringSpec,
    img_dir: Path,
) -> None:
    """Generate the 4 README Matplotlib figures in docs/img/*.png."""
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()

    BLUE = "#2a78d6"
    NAVY = "#1b3f8f"
    ORANGE = "#eb6834"
    WARM = "#d99026"
    GREEN = "#1baf7a"
    RED = "#c8423b"
    PURPLE = "#7a52b3"
    GREY = "#7d7b74"
    INK = "#1f1f1d"

    # -------------------------------------------------------------------------
    # 1. docs/img/iv_pv_curves.png
    # -------------------------------------------------------------------------
    fig, (ax_iv, ax_pv) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    _style_ax(ax_iv)
    _style_ax(ax_pv)

    cases = [
        ("Healthy string (72 sub-strings)", SubstringFaultState(), INK, "-"),
        ("2 shorted bypass diodes (-28.6 V)", SubstringFaultState(n_shorted_bypass_diodes=2), BLUE, "-"),
        ("Corroded MC4 (+2.5 Ohm R_s)", SubstringFaultState(extra_rs_ohm=2.5), ORANGE, "--"),
        ("PID shunt loss (4 modules, R_sh=35 Ohm)", SubstringFaultState(n_pid_modules=4, pid_rsh_module_ohm=35.0), PURPLE, "-."),
        ("Partial shading (6 sub-strings @ 35 % G)", SubstringFaultState(n_shaded_substrings=6, shaded_transmission_frac=0.35), GREEN, "-"),
    ]

    for label, f_state, col, ls in cases:
        res = sweep_string_iv(g_poa_wm2=900.0, t_cell_c=35.0, fault=f_state, spec=spec, n_points=500)
        v_arr = res["v_v"]
        i_arr = res["i_a"]
        p_kw = res["p_w"] / 1000.0
        v_mp = float(res["v_mppt_v"])
        i_mp = float(res["i_mppt_a"])
        p_mp = float(res["p_mppt_w"]) / 1000.0

        ax_iv.plot(v_arr, i_arr, label=label, color=col, ls=ls, lw=1.5)
        ax_iv.plot(v_mp, i_mp, marker="o", color=col, markersize=5)

        ax_pv.plot(v_arr, p_kw, label=label, color=col, ls=ls, lw=1.5)
        ax_pv.plot(v_mp, p_mp, marker="o", color=col, markersize=5)

        if f_state.n_shaded_substrings > 0 and abs(float(res["v_local_v"]) - v_mp) > 30.0:
            v_loc = float(res["v_local_v"])
            p_loc = float(res["p_local_w"]) / 1000.0
            ax_pv.plot(v_loc, p_loc, marker="s", color=RED, markersize=5.5, label="Shaded local false peak")
            ax_pv.annotate(
                "Global MPPT",
                xy=(v_mp, p_mp),
                xytext=(v_mp - 210, p_mp - 2.3),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GREEN),
            )
            ax_pv.annotate(
                "Local false peak",
                xy=(v_loc, p_loc),
                xytext=(v_loc - 330, p_loc - 1.2),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=RED),
            )

    ax_iv.set_xlim(0, 1220)
    ax_iv.set_ylim(0, 13.8)
    ax_iv.set_xlabel("String voltage V [V]")
    ax_iv.set_ylabel("String current I [A]")
    ax_iv.set_title("String I-V curves (G = 900 W/m2, Tc = 35 degC)", fontsize=9.5)
    ax_iv.grid(alpha=0.25)
    ax_iv.legend(loc="lower left", frameon=True, fontsize=7.5)

    ax_pv.set_xlim(0, 1220)
    ax_pv.set_ylim(0, 13.8)
    ax_pv.set_xlabel("String voltage V [V]")
    ax_pv.set_ylabel("String power P [kW]")
    ax_pv.set_title("String P-V curves and MPPT operating points", fontsize=9.5)
    ax_pv.grid(alpha=0.25)
    ax_pv.legend(loc="upper left", frameon=True, fontsize=7.5)

    _save(fig, img_dir / "iv_pv_curves.png")

    # -------------------------------------------------------------------------
    # 2. docs/img/dv_vs_current_signature.png
    # -------------------------------------------------------------------------
    fig, (ax_dv, ax_pid) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    _style_ax(ax_dv)
    _style_ax(ax_pid)

    g_arr = df_15m["g_poa_wm2"].to_numpy()
    doy_arr = df_15m["doy"].to_numpy()
    v_peer = df_15m["v_peer_v"].to_numpy()
    mask_post = (doy_arr >= 215) & (g_arr > 120.0) & (v_peer > 500.0)
    idx_sub = np.where(mask_post)[0][::8]

    sig_channels = [
        ("mppt_7", "1 shorted diode (MPPT 7, ~14.3 V)", BLUE, "o"),
        ("mppt_2", "2 shorted diodes (MPPT 2, ~28.6 V)", NAVY, "o"),
        ("mppt_8", "4 shorted diodes (MPPT 8, ~57.2 V)", PURPLE, "o"),
        ("mppt_9", "+1.5 Ohm R_s (MPPT 9)", WARM, "^"),
        ("mppt_3", "+2.5 Ohm R_s (MPPT 3)", ORANGE, "^"),
        ("mppt_10", "+3.2 Ohm R_s (MPPT 10)", RED, "^"),
    ]
    for slug, lbl, col, mk in sig_channels:
        i_pts = df_15m[f"i_{slug}"].to_numpy()[idx_sub]
        dv_pts = (v_peer - df_15m[f"v_{slug}"].to_numpy())[idx_sub]
        ax_dv.scatter(i_pts, dv_pts, s=9, alpha=0.45, color=col, marker=mk, label=lbl)

    i_line = np.linspace(1.0, 13.2, 100)
    for n_b, col in [(1, BLUE), (2, NAVY), (4, PURPLE)]:
        ax_dv.axhline(n_b * 14.3, color=col, ls=":", lw=1.0)
    for rs_v, col in [(1.5, WARM), (2.5, ORANGE), (3.2, RED)]:
        ax_dv.plot(i_line, rs_v * 0.989 * i_line, color=col, ls="--", lw=1.1)

    ax_dv.set_xlim(1.0, 13.5)
    ax_dv.set_ylim(-3.0, 68.0)
    ax_dv.set_xlabel("MPPT string current I_mppt [A]")
    ax_dv.set_ylabel("Peer voltage deficit Delta V_k [V]")
    ax_dv.set_title("Voltage deficit Delta V_k vs string current I_mppt", fontsize=9.5)
    ax_dv.grid(alpha=0.25)
    ax_dv.legend(loc="upper left", frameon=True, fontsize=7.2, ncol=1)

    # Right panel: normalised voltage ratio v_k vs irradiance G_poa (PID low-G droop)
    g_bins = np.linspace(120.0, 1000.0, 12)
    g_centers = 0.5 * (g_bins[:-1] + g_bins[1:])
    pid_compare = [
        ("mppt_1", "Healthy (MPPT 1)", INK, "-"),
        ("mppt_6", "Uniform soiling 4.5 % (MPPT 6)", GREEN, "--"),
        ("mppt_2", "2 shorted diodes (MPPT 2)", BLUE, "-."),
        ("mppt_3", "Corroded MC4 +2.5 Ohm (MPPT 3)", ORANGE, ":"),
        ("mppt_4", "PID shunt loss R_sh=35 Ohm (MPPT 4)", PURPLE, "-"),
    ]
    for slug, lbl, col, ls in pid_compare:
        v_ratio_all = df_15m[f"v_{slug}"].to_numpy() / np.maximum(v_peer, 1.0)
        meds = []
        for b0, b1 in zip(g_bins[:-1], g_bins[1:]):
            bm = mask_post & (g_arr >= b0) & (g_arr < b1)
            meds.append(float(np.median(v_ratio_all[bm])) if np.any(bm) else np.nan)
        ax_pid.plot(g_centers, meds, label=lbl, color=col, ls=ls, marker="o", markersize=3.5, lw=1.4)

    ax_pid.axvspan(100, 300, color=PURPLE, alpha=0.08, label="Low-G PID window (<300 W/m2)")
    ax_pid.set_xlim(100, 1020)
    ax_pid.set_ylim(0.948, 1.006)
    ax_pid.set_xlabel("POA irradiance G_poa [W/m2]")
    ax_pid.set_ylabel("Normalised voltage ratio v_k = V_k / V_peer [-]")
    ax_pid.set_title("Low-irradiance PID droop vs diode and R_s faults", fontsize=9.5)
    ax_pid.grid(alpha=0.25)
    ax_pid.legend(loc="lower right", frameon=True, fontsize=7.2)

    _save(fig, img_dir / "dv_vs_current_signature.png")

    # -------------------------------------------------------------------------
    # 3. docs/img/mppt_fleet_timeline.png
    # -------------------------------------------------------------------------
    fig, (ax_t1, ax_t2) = plt.subplots(2, 1, figsize=(10.2, 5.2), sharex=True)
    _style_ax(ax_t1)
    _style_ax(ax_t2)

    doy_x = df_daily["doy"].to_numpy()
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

    ax_t1.plot(doy_x, df_daily["dv_med_mppt_1"], label="MPPT 1: Healthy", color=INK, lw=1.0)
    ax_t1.plot(doy_x, df_daily["dv_med_mppt_7"], label="MPPT 7: 1 shorted diode (Apr)", color=BLUE, lw=1.2)
    ax_t1.plot(doy_x, df_daily["dv_med_mppt_2"], label="MPPT 2: 2 shorted diodes (May)", color=NAVY, lw=1.3)
    ax_t1.plot(doy_x, df_daily["dv_med_mppt_8"], label="MPPT 8: 4 shorted diodes (Jun)", color=PURPLE, lw=1.3)
    ax_t1.plot(doy_x, df_daily["dv_med_mppt_3"], label="MPPT 3: +2.5 Ohm R_s ramp (Jul)", color=ORANGE, lw=1.2)
    ax_t1.plot(doy_x, df_daily["dv_med_mppt_4"], label="MPPT 4: PID shunt loss (Jun)", color=RED, ls="--", lw=1.1)
    ax_t1.set_ylabel("Daily median Delta V_k [V]")
    ax_t1.set_ylim(-4, 65)
    ax_t1.set_xlim(1, 365)
    ax_t1.set_title("12-month daily MPPT tracking: step drops (bypass diodes), R_s ramp, PID, and rain-washed soiling", fontsize=9.5)
    ax_t1.grid(alpha=0.25)
    ax_t1.legend(loc="upper left", frameon=True, fontsize=7.5, ncol=3)

    ax_t2.plot(doy_x, df_daily["i_norm_mppt_1"], label="MPPT 1: Healthy (i_k ~ 1.000)", color=INK, lw=1.0)
    ax_t2.plot(doy_x, df_daily["i_norm_mppt_2"], label="MPPT 2: 2 shorted diodes (i_k unchanged)", color=NAVY, ls="--", lw=1.0)
    ax_t2.plot(doy_x, df_daily["i_norm_mppt_6"], label="MPPT 6: Uniform soiling (rain-washed steps)", color=GREEN, lw=1.3)
    ax_t2.plot(doy_x, df_daily["i_norm_mppt_4"], label="MPPT 4: PID shunt degradation", color=RED, ls=":", lw=1.1)
    ax_t2.set_xticks(month_starts[::2], month_labels[::2])
    ax_t2.set_ylabel("Daily median current ratio i_k [-]")
    ax_t2.set_xlabel("Date (2025)")
    ax_t2.set_ylim(0.935, 1.015)
    ax_t2.grid(alpha=0.25)
    ax_t2.legend(loc="lower left", frameon=True, fontsize=7.5, ncol=2)

    _save(fig, img_dir / "mppt_fleet_timeline.png")

    # -------------------------------------------------------------------------
    # 4. docs/img/verification_recovery.png
    # -------------------------------------------------------------------------
    fig, (ax_p1, ax_p2, ax_p3) = plt.subplots(1, 3, figsize=(10.8, 3.7))
    _style_ax(ax_p1)
    _style_ax(ax_p2)
    _style_ax(ax_p3)

    # Panel 1: True vs recovered Delta R_s [Ohm] and shorted diode voltage intercept / 10 [10 V]
    rs_rec = verification["series_resistance_recovery"]
    bd_rec = verification["bypass_diode_recovery"]

    ax_p1.plot([0, 6.2], [0, 6.2], color=GREY, ls="--", lw=1.0, label="1:1 parity")
    ax_p1.scatter(
        [r["rs_true_ohm"] for r in rs_rec],
        [r["rs_est_ohm"] for r in rs_rec],
        color=ORANGE,
        s=45,
        marker="^",
        zorder=4,
        label="Delta R_s [Ohm]",
    )
    ax_p1.scatter(
        [b["dv_true_v"] / 10.0 for b in bd_rec],
        [b["dv_est_v"] / 10.0 for b in bd_rec],
        color=BLUE,
        s=45,
        marker="o",
        zorder=4,
        label="Diode step Delta V / 10 [10 V]",
    )
    ax_p1.set_xlim(0, 6.2)
    ax_p1.set_ylim(0, 6.2)
    ax_p1.set_xlabel("True parameter [Ohm or 10 V]")
    ax_p1.set_ylabel("Estimated parameter [Ohm or 10 V]")
    ax_p1.set_title("Parameter recovery parity", fontsize=9.5)
    ax_p1.grid(alpha=0.25)
    ax_p1.legend(loc="upper left", frameon=True, fontsize=7.5)

    # Panel 2: True vs estimated lost annual energy [MWh] per faulted channel
    fault_df = df_diag[df_diag["true_class"] != "Healthy"]
    max_e = max(float(fault_df["lost_energy_true_mwh"].max()), float(fault_df["lost_energy_est_mwh"].max())) * 1.15
    ax_p2.plot([0, max_e], [0, max_e], color=GREY, ls="--", lw=1.0, label="1:1 parity")
    ax_p2.scatter(
        fault_df["lost_energy_true_mwh"],
        fault_df["lost_energy_est_mwh"],
        color=NAVY,
        s=42,
        zorder=4,
        label="Faulted MPPT channels",
    )
    ax_p2.set_xlim(0, max_e)
    ax_p2.set_ylim(0, max_e)
    ax_p2.set_xlabel("True lost energy [MWh / yr]")
    ax_p2.set_ylabel("Estimated lost energy [MWh / yr]")
    ax_p2.set_title("Annual energy loss recovery", fontsize=9.5)
    ax_p2.grid(alpha=0.25)
    ax_p2.legend(loc="upper left", frameon=True, fontsize=7.5)

    # Panel 3: Per-class F1 score (no text above bars!)
    short_names = ["Healthy", "Bypass\ndiode", "High R_s", "PID\nlow R_sh", "Partial\nshading", "Uniform\nsoiling"]
    f1_vals = [verification["class_metrics"][c]["f1"] for c in FAULT_CLASSES]
    x_pos = np.arange(len(FAULT_CLASSES))
    ax_p3.bar(x_pos, f1_vals, width=0.55, color=BLUE, label="Class F1 score")
    ax_p3.axhline(
        verification["macro_f1"],
        color=RED,
        ls="--",
        lw=1.1,
        label=f"Macro-F1 = {verification['macro_f1']:.3f}",
    )
    ax_p3.set_xticks(x_pos, short_names, fontsize=7.5)
    ax_p3.set_ylim(0.0, 1.12)
    ax_p3.set_ylabel("F1 score [-]")
    ax_p3.set_title("6-class fault classification F1", fontsize=9.5)
    ax_p3.grid(axis="y", alpha=0.25)
    ax_p3.set_axisbelow(True)
    ax_p3.legend(loc="lower right", frameon=True, fontsize=7.5)

    _save(fig, img_dir / "verification_recovery.png")


def build_dashboard_payload(
    df_15m: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_diag: pd.DataFrame,
    verification: dict,
    spec: InverterStringSpec,
) -> dict:
    """Build compact JSON payload for docs/data.js powering the interactive Plotly dashboard."""
    g_arr = df_15m["g_poa_wm2"].to_numpy()
    doy_arr = df_15m["doy"].to_numpy()
    v_peer = df_15m["v_peer_v"].to_numpy()
    i_peer = df_15m["i_peer_a"].to_numpy()
    mask_post = (doy_arr >= 215) & (g_arr > 110.0) & (v_peer > 450.0)
    sample_idx = np.where(mask_post)[0][::12]

    channels_payload = {}
    fault_states_map = {
        "MPPT 1": SubstringFaultState(),
        "MPPT 2": SubstringFaultState(n_shorted_bypass_diodes=2),
        "MPPT 3": SubstringFaultState(extra_rs_ohm=2.5),
        "MPPT 4": SubstringFaultState(n_pid_modules=4, pid_rsh_module_ohm=35.0),
        "MPPT 5": SubstringFaultState(n_shaded_substrings=6, shaded_transmission_frac=0.35),
        "MPPT 6": SubstringFaultState(soiling_loss_frac=0.045),
        "MPPT 7": SubstringFaultState(n_shorted_bypass_diodes=1),
        "MPPT 8": SubstringFaultState(n_shorted_bypass_diodes=4),
        "MPPT 9": SubstringFaultState(extra_rs_ohm=1.5),
        "MPPT 10": SubstringFaultState(extra_rs_ohm=3.2),
        "MPPT 11": SubstringFaultState(),
        "MPPT 12": SubstringFaultState(),
    }

    for idx, cfg in enumerate(MPPT_CHANNEL_CONFIGS, start=1):
        ch = cfg["channel"]
        slug = f"mppt_{idx}"
        f_state = fault_states_map[ch]
        iv_res = sweep_string_iv(900.0, 35.0, f_state, spec=spec, n_points=160)
        diag_row = df_diag[df_diag["channel"] == ch].iloc[0].to_dict()

        i_pts = df_15m[f"i_{slug}"].to_numpy()[sample_idx]
        dv_pts = (v_peer - df_15m[f"v_{slug}"].to_numpy())[sample_idx]
        g_pts = g_arr[sample_idx]
        v_norm_pts = (df_15m[f"v_{slug}"].to_numpy() / np.maximum(v_peer, 1.0))[sample_idx]

        channels_payload[ch] = {
            "channel": ch,
            "label": cfg["label"],
            "description": cfg["description"],
            "fault_class": cfg["fault_class"],
            "diagnostics": diag_row,
            "iv_curve_900wm2": {
                "v_v": [round(float(x), 1) for x in iv_res["v_v"]],
                "i_a": [round(float(x), 3) for x in iv_res["i_a"]],
                "p_kw": [round(float(x) / 1000.0, 3) for x in iv_res["p_w"]],
                "v_mppt_v": round(float(iv_res["v_mppt_v"]), 1),
                "i_mppt_a": round(float(iv_res["i_mppt_a"]), 2),
                "p_mppt_kw": round(float(iv_res["p_mppt_w"]) / 1000.0, 2),
                "v_local_v": round(float(iv_res["v_local_v"]), 1),
                "p_local_kw": round(float(iv_res["p_local_w"]) / 1000.0, 2),
            },
            "scatter_dv_i": {
                "i_a": [round(float(x), 2) for x in i_pts],
                "dv_v": [round(float(x), 2) for x in dv_pts],
                "g_wm2": [round(float(x), 0) for x in g_pts],
                "v_norm": [round(float(x), 4) for x in v_norm_pts],
            },
            "daily": {
                "date": df_daily["date"].tolist(),
                "v_norm": [round(float(x), 4) for x in df_daily[f"v_norm_{slug}"]],
                "i_norm": [round(float(x), 4) for x in df_daily[f"i_norm_{slug}"]],
                "dv_med": [round(float(x), 2) for x in df_daily[f"dv_med_{slug}"]],
            },
        }

    return {
        "spec": {
            "n_mppt": spec.n_mppt,
            "strings_per_mppt": spec.strings_per_mppt,
            "n_modules_per_string": spec.n_modules_per_string,
            "substrings_per_module": spec.substrings_per_module,
            "n_substrings": spec.n_substrings,
            "p_mp_module_stc_w": spec.p_mp_module_stc_w,
            "v_mp_module_stc_v": spec.v_mp_module_stc_v,
            "v_mp_string_stc_v": round(spec.v_mp_string_stc_v, 1),
            "i_mp_stc_a": spec.i_mp_stc_a,
            "v_oc_string_stc_v": round(spec.v_oc_string_stc_v, 1),
            "i_sc_stc_a": spec.i_sc_stc_a,
            "step_voltage_per_shorted_bd_stc_v": round(spec.step_voltage_per_shorted_bd_stc_v, 2),
        },
        "verification": verification,
        "fleet_table": df_diag.to_dict(orient="records"),
        "channels": channels_payload,
    }


def run_full_pipeline(output_root: Path | None = None) -> dict:
    """Execute simulation, fault diagnosis, verification, figure generation, and dashboard export."""
    root = output_root or REPO_ROOT
    results_dir = root / "results"
    docs_dir = root / "docs"
    img_dir = docs_dir / "img"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    spec = InverterStringSpec()
    df_15m, df_daily = simulate_annual_mppt_dataset(spec=spec, seed=42)
    df_diag = diagnose_mppt_channels(df_15m, spec=spec)
    verification = evaluate_fault_verification(df_15m, df_diag, spec=spec)

    df_diag.to_csv(results_dir / "mppt_channel_diagnostics.csv", index=False)
    df_daily.to_csv(results_dir / "daily_fleet_tracking.csv", index=False)
    (results_dir / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")

    generate_plots(df_15m, df_daily, df_diag, verification, spec, img_dir)

    payload = build_dashboard_payload(df_15m, df_daily, df_diag, verification, spec)
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")
    (docs_dir / "data.js").write_text(
        "window.DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )

    return {
        "diagnostics": df_diag,
        "verification": verification,
        "payload": payload,
    }
