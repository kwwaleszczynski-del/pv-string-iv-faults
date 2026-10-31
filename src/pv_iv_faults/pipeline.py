"""End-to-end pipeline: tune on one set of synthetic inverters, report on held-out ones, showcase, figures.

1. Simulate the tuning inverters (seeds 1001-1008) and fit the detector thresholds on their
   fault-free windows only.
2. Freeze the thresholds and score 14-day windows of the held-out inverters (seeds 2001-2008),
   whose sites, weather, fault types, sizes and onsets are drawn independently.
3. Run the same frozen estimator on the hand-written 12-channel showcase inverter (seed 42) for the
   channel table, the figures and the interactive page.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .diagnostics import (  # noqa: E402
    DETECTORS,
    FAULT_CLASSES,
    NO_DECISION,
    channel_features,
    classify,
    daily_tracking,
    estimated_loss_mwh,
    fit_thresholds,
    prepare_samples,
    window_features,
)
from .evaluation import MIN_IMPACT_PCT, attach_truth, classification_scores, detection_delays  # noqa: E402
from .model import InverterStringSpec, SubstringFaultState, cell_temperature_c, sweep_string_iv  # noqa: E402
from .simulator import SHOWCASE_SCENARIOS, simulate_inverter, simulate_random_inverter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TUNING_SEEDS: tuple[int, ...] = tuple(range(1001, 1009))
HELDOUT_SEEDS: tuple[int, ...] = tuple(range(2001, 2009))
SHOWCASE_SEED = 42
SHOWCASE_DAYS = (213, 273)  # 1 Aug - 30 Sep: the "diagnose now" period of the showcase channel table
THRESHOLD_QUANTILE = 0.99
SHORT = {
    "Healthy": "Healthy",
    "Shorted Bypass Diode": "Bypass diode",
    "High Series Resistance R_s": "High R_s",
    "PID / Low R_sh": "PID",
    "Partial Shading": "Shading",
    "Uniform Soiling": "Soiling",
    NO_DECISION: "No decision",
}


def inverter_windows(seed: int) -> tuple[pd.DataFrame, list]:
    """Window features plus ground truth of one random synthetic inverter."""
    res = simulate_random_inverter(seed)
    s = prepare_samples(res.telemetry, res.site.lat_deg, res.site.lon_deg)
    windows = attach_truth(window_features(s), res)
    windows.insert(0, "seed", seed)
    return windows, res.scenarios


def _run_seeds(seeds: tuple[int, ...], workers: int) -> list[tuple[pd.DataFrame, list]]:
    if workers <= 1:
        return [inverter_windows(seed) for seed in seeds]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(inverter_windows, seeds))


def _split_report(pred: pd.DataFrame, scenarios_by_seed: dict[int, list]) -> dict:
    """Scores of one split: classification, excluded windows, parameter recovery, detection delay."""
    scored = pred[pred["true_class"].isin(FAULT_CLASSES)]
    scores = classification_scores(scored["true_class"].tolist(), scored["predicted_class"].tolist())
    healthy = scored[scored["true_class"] == "Healthy"]
    healthy_decided = healthy[healthy["predicted_class"] != NO_DECISION]
    faults_flagged = lambda df: int((~df["predicted_class"].isin(["Healthy", NO_DECISION])).sum())
    # operational reading: an inverter whose channels are all healthy, undecided windows count as no alarm
    n_channels, n_windows = pred["channel"].nunique(), pred["window"].nunique()
    fa_per_inverter_window = n_channels * faults_flagged(healthy) / max(len(healthy), 1)
    healthy_fa = healthy[~healthy["predicted_class"].isin(["Healthy", NO_DECISION])]

    excluded = {}
    for lbl in ("Transition", "Below impact"):
        sub = pred[pred["true_class"] == lbl]
        excluded[lbl] = {"windows": len(sub), "flagged_as_a_fault": faults_flagged(sub)}

    def hits(cls: str) -> pd.DataFrame:
        return pred[(pred["true_class"] == cls) & (pred["predicted_class"] == cls)]

    bd, rs, soil = hits("Shorted Bypass Diode"), hits("High Series Resistance R_s"), hits("Uniform Soiling")
    rs_err = 100.0 * (rs["rs_est_ohm"] - rs["rs_true_ohm"]) / rs["rs_true_ohm"]
    soil_err = soil["soiling_est_pct"] - soil["soiling_true_pct"]
    delays = detection_delays(pred, scenarios_by_seed)
    delay_by_class = {}
    for cls in FAULT_CLASSES[1:]:
        d = delays[delays["fault_class"] == cls]["delay_days"]
        delay_by_class[cls] = {
            "faulted_channels": int(len(d)),
            "detected": int(d.notna().sum()),
            "median_delay_days": None if d.notna().sum() == 0 else float(d.median()),
        }

    return {
        "inverters": len(scenarios_by_seed),
        "channel_windows": len(pred),
        "scored_windows": len(scored),
        "decided_pct": round(100.0 * float((scored["predicted_class"] != NO_DECISION).mean()), 1),
        "healthy_false_alarm_pct": round(100.0 * faults_flagged(healthy_decided) / max(len(healthy_decided), 1), 1),
        "false_alarms_per_inverter_window": round(fa_per_inverter_window, 2),
        "false_alarms_per_inverter_year": round(fa_per_inverter_window * n_windows, 1),
        "healthy_channels": int(healthy.groupby(["seed", "channel"]).ngroups),
        "healthy_channels_with_false_alarm": int(healthy_fa.groupby(["seed", "channel"]).ngroups),
        **scores,
        "excluded_windows": excluded,
        "bypass_diode_count": {
            "windows": len(bd),
            "exact_pct": round(100.0 * float((bd["n_bd_est"] == bd["n_bd_true"]).mean()), 1) if len(bd) else None,
            "table": {
                int(n): {int(e): int(c) for e, c in grp["n_bd_est"].value_counts().sort_index().items()}
                for n, grp in bd.groupby("n_bd_true")
            },
        },
        "series_resistance": {
            "windows": len(rs),
            "rel_error_pct_median": round(float(rs_err.median()), 1) if len(rs) else None,
            "rel_error_pct_p10": round(float(rs_err.quantile(0.1)), 1) if len(rs) else None,
            "rel_error_pct_p90": round(float(rs_err.quantile(0.9)), 1) if len(rs) else None,
        },
        "soiling": {
            "windows": len(soil),
            "error_pp_median": round(float(soil_err.median()), 2) if len(soil) else None,
            "abs_error_pp_median": round(float(soil_err.abs().median()), 2) if len(soil) else None,
        },
        "detection": delay_by_class,
    }


def _showcase(spec: InverterStringSpec, thresholds: dict[str, float]) -> dict:
    """Run the frozen estimator on the hand-written showcase inverter and join its ground truth."""
    res = simulate_inverter(SHOWCASE_SCENARIOS, seed=SHOWCASE_SEED, spec=spec)
    s = prepare_samples(res.telemetry, res.site.lat_deg, res.site.lon_deg)
    year = classify(attach_truth(window_features(s), res), thresholds, spec)

    d0, d1 = SHOWCASE_DAYS
    rows = [{"channel": f"MPPT {k + 1}", "window": -1, "day_from": d0, "day_to": d1, **channel_features(s, k, d0, d1)} for k in range(spec.n_mppt)]
    table = classify(attach_truth(pd.DataFrame(rows), res), thresholds, spec)
    tr = res.truth
    est_loss = estimated_loss_mwh(s)
    table.insert(1, "label", [sc.label for sc in res.scenarios])
    table.insert(2, "injected_fault", [sc.fault_class for sc in res.scenarios])
    table["annual_energy_mwh"] = [round(tr[f"p_true_mppt_{k}"].sum() * 0.25 / 1e6, 2) for k in range(1, spec.n_mppt + 1)]
    table["lost_energy_true_mwh"] = [
        round((tr[f"p_counterfactual_mppt_{k}"] - tr[f"p_true_mppt_{k}"]).sum() * 0.25 / 1e6, 3) for k in range(1, spec.n_mppt + 1)
    ]
    table["lost_energy_est_mwh"] = np.round(est_loss, 3)
    scenarios = {SHOWCASE_SEED: res.scenarios}
    year.insert(0, "seed", SHOWCASE_SEED)
    return {"result": res, "samples": s, "year": year, "table": table, "report": _split_report(year, scenarios)}


def _fmt(x, d=2) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def summary_markdown(ver: dict, table: pd.DataFrame) -> str:
    """Markdown tables quoted verbatim in README.md (a test checks they stay identical)."""
    h, t = ver["heldout"], ver["tuning"]
    lines = ["<!-- generated by python -m pv_iv_faults: results/summary.md -->", ""]
    lines += [
        "| Class | Held-out windows | Precision | Recall | F1 | Tuning F1 (in-sample) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in FAULT_CLASSES:
        m = h["class_metrics"][c]
        lines.append(f"| {c} | {m['support']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {t['class_metrics'][c]['f1']:.3f} |")
    lines.append(f"| **Macro-F1** | {h['scored_windows']} | | | **{h['macro_f1']:.3f}** | {t['macro_f1']:.3f} |")
    lines += [
        "",
        f"Held-out: {h['inverters']} inverters x 12 channels x 26 windows; {h['decided_pct']:.1f} % of scored windows had enough "
        f"irradiance for a decision; {h['healthy_false_alarm_pct']:.1f} % of decided healthy windows raised a fault alarm. "
        f"In operation, an inverter with all 12 channels healthy would get {h['false_alarms_per_inverter_window']:.2f} false "
        f"alarms per 14-day window, about {h['false_alarms_per_inverter_year']:.0f} a year; they came from "
        f"{h['healthy_channels_with_false_alarm']} of the {h['healthy_channels']} healthy channels. "
        f"Excluded from scoring: {h['excluded_windows']['Transition']['windows']} onset windows and "
        f"{h['excluded_windows']['Below impact']['windows']} windows whose fault cost less than {MIN_IMPACT_PCT:.1f} % of the "
        f"window's energy ({h['excluded_windows']['Below impact']['flagged_as_a_fault']} of them were still flagged).",
        "",
        "| Parameter (held-out, correctly classified windows) | Windows | Result |",
        "|---|---:|---|",
        f"| Shorted bypass diode count $`\\hat{{n}}_{{\\text{{bd}}}}`$ | {h['bypass_diode_count']['windows']} | "
        f"{_fmt(h['bypass_diode_count']['exact_pct'], 1)} % exact |",
        f"| Series resistance $`\\widehat{{\\Delta R_s}}`$ relative error | {h['series_resistance']['windows']} | "
        f"median {_fmt(h['series_resistance']['rel_error_pct_median'], 1)} %, 10-90 % range "
        f"{_fmt(h['series_resistance']['rel_error_pct_p10'], 1)} % to {_fmt(h['series_resistance']['rel_error_pct_p90'], 1)} % |",
        f"| Soiling current loss, estimate minus truth | {h['soiling']['windows']} | "
        f"median {_fmt(h['soiling']['error_pp_median'])} pp, median absolute {_fmt(h['soiling']['abs_error_pp_median'])} pp |",
        "",
        "| Fault family (held-out) | Faulted channels | Detected | Median days from onset to first correct window |",
        "|---|---:|---:|---:|",
    ]
    for c in FAULT_CLASSES[1:]:
        d = h["detection"][c]
        lines.append(f"| {c} | {d['faulted_channels']} | {d['detected']} | {_fmt(d['median_delay_days'], 0)} |")
    lines += ["", "| Detector (rule order) | Feature | Threshold (99th percentile of healthy tuning windows) |", "|---|---|---:|"]
    for cls, feat in DETECTORS:
        val = ver["thresholds"][feat]
        shown = f"{val:.3f} Ohm" if feat == "dv_slope_ohm" else f"{100.0 * val:.2f} %"
        lines.append(f"| {cls} | `{feat}` | {shown} |")

    d0, d1 = SHOWCASE_DAYS
    lines += [
        "",
        f"Showcase inverter, diagnosed on days {d0}-{d1} (1 Aug - 30 Sep):",
        "",
        "| Channel | Injected fault | Diagnosed | True $`n_{\\text{bd}}`$ | Est. $`\\hat{n}_{\\text{bd}}`$ | True $`\\Delta R_s`$ [Ohm] | "
        "Est. $`\\widehat{\\Delta R_s}`$ [Ohm] | True soiling [%] | Est. soiling [%] | Low-G power deficit [%] | Annual energy [MWh] | "
        "True loss [MWh] | Est. loss [MWh] |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['label']} | {r['injected_fault']} | {r['predicted_class']} | {int(r['n_bd_true'])} | {int(r['n_bd_est'])} | "
            f"{r['rs_true_ohm']:.2f} | {r['rs_est_ohm']:.2f} | {r['soiling_true_pct']:.2f} | {r['soiling_est_pct']:.2f} | "
            f"{100.0 * r['p_def_lo']:.2f} | {r['annual_energy_mwh']:.2f} | {r['lost_energy_true_mwh']:.3f} | {r['lost_energy_est_mwh']:.3f} |"
        )
    s = ver["showcase"]
    lines += [
        "",
        f"Showcase, all 26 windows of the year: macro-F1 {s['macro_f1']:.3f}, {s['decided_pct']:.1f} % decided, "
        f"PID first flagged {_fmt(s['detection']['PID / Low R_sh']['median_delay_days'], 0)} days after onset.",
        "",
    ]
    return "\n".join(lines)


def generate_plots(sc: dict, ver: dict, heldout: pd.DataFrame, spec: InverterStringSpec, img_dir: Path) -> None:
    """Generate the 4 README figures in docs/img/*.png."""
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    BLUE, NAVY, ORANGE, WARM = "#2a78d6", "#1b3f8f", "#eb6834", "#d99026"
    GREEN, RED, PURPLE, GREY, INK = "#1baf7a", "#c8423b", "#7a52b3", "#7d7b74", "#1f1f1d"

    def style(ax):
        for spine in ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.8)
        ax.grid(alpha=0.25)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(img_dir / name, dpi=140, bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)

    # 1. Model I-V and P-V curves --------------------------------------------------------------
    fig, (ax_iv, ax_pv) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    base = sweep_string_iv(900.0, 35.0, SubstringFaultState(), spec=spec, n_points=500)
    step = base["v_mppt_v"] - sweep_string_iv(900.0, 35.0, SubstringFaultState(n_shorted_bypass_diodes=2), spec=spec)["v_mppt_v"]
    cases = [
        ("Healthy string (72 sub-strings)", SubstringFaultState(), INK, "-"),
        (f"2 shorted bypass diodes ({-step:.1f} V)", SubstringFaultState(n_shorted_bypass_diodes=2), BLUE, "-"),
        ("+2.5 Ohm series resistance", SubstringFaultState(extra_rs_ohm=2.5), ORANGE, "--"),
        ("PID, 4 modules at R_sh = 35 Ohm", SubstringFaultState(n_pid_modules=4, pid_rsh_module_ohm=35.0), PURPLE, "-."),
        ("6 sub-strings at 15 % irradiance", SubstringFaultState(n_shaded_substrings=6, shaded_transmission_frac=0.15), GREEN, "-"),
    ]
    for label, state, col, ls in cases:
        r = sweep_string_iv(900.0, 35.0, state, spec=spec, n_points=500)
        ax_iv.plot(r["v_v"], r["i_a"], label=label, color=col, ls=ls, lw=1.5)
        ax_iv.plot(r["v_mppt_v"], r["i_mppt_a"], marker="o", color=col, markersize=5)
        ax_pv.plot(r["v_v"], r["p_w"] / 1000.0, label=label, color=col, ls=ls, lw=1.5)
        ax_pv.plot(r["v_mppt_v"], r["p_mppt_w"] / 1000.0, marker="o", color=col, markersize=5)
        if state.n_shaded_substrings and abs(r["v_local_v"] - r["v_mppt_v"]) > 30.0:
            ax_pv.plot(r["v_local_v"], r["p_local_w"] / 1000.0, marker="s", color=RED, markersize=5.5, label="Local (non-global) peak")
    for ax, ylab, title in [
        (ax_iv, "String current I [A]", "Model I-V curves (G = 900 W/m2, Tc = 35 degC)"),
        (ax_pv, "String power P [kW]", "Model P-V curves, dots = global MPP"),
    ]:
        style(ax)
        ax.set_xlim(0, 1220)
        ax.set_ylim(0, 13.8)
        ax.set_xlabel("String voltage V [V]")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
    ax_iv.legend(loc="lower left", fontsize=7.5)
    ax_pv.legend(loc="upper left", fontsize=7.5)
    save(fig, "iv_pv_curves.png")

    # 2. Diagnostic plane and irradiance-binned power deficit (showcase, Aug-Sep) ---------------
    s, table = sc["samples"], sc["table"]
    d0, d1 = SHOWCASE_DAYS
    fig, (ax_dv, ax_g) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    style(ax_dv)
    style(ax_g)
    m = (s.doy >= d0) & (s.doy <= d1) & (s.g > 250.0) & (s.v_peer > 100.0)
    idx = np.flatnonzero(m)[::3]
    i_line = np.linspace(2.0, 13.0, 50)
    for k, col, mk in [(7, BLUE, "o"), (2, NAVY, "o"), (8, PURPLE, "o"), (9, WARM, "^"), (3, ORANGE, "^"), (10, RED, "^")]:
        row = table.iloc[k - 1]
        ax_dv.scatter(s.i[idx, k - 1], (s.v_peer - s.v[:, k - 1])[idx], s=8, alpha=0.4, color=col, marker=mk, label=row["label"].split(": ", 1)[1])
        ax_dv.plot(i_line, row["dv_intercept_v"] + row["dv_slope_ohm"] * i_line, color=col, lw=1.1)
    ax_dv.set_xlabel("MPPT current I_k [A]")
    ax_dv.set_ylabel("Peer voltage deficit V_peer - V_k [V]")
    ax_dv.set_title("Diagnostic plane, Aug-Sep (lines = robust fits)", fontsize=9.5)
    ax_dv.legend(loc="upper left", fontsize=6.8)

    g_edges = np.arange(100.0, 1051.0, 75.0)
    g_mid = 0.5 * (g_edges[:-1] + g_edges[1:])
    for k, col, ls in [(1, INK, "-"), (6, GREEN, "--"), (2, BLUE, "-."), (3, ORANGE, ":"), (4, PURPLE, "-")]:
        mk = (s.doy >= d0) & (s.doy <= d1) & (s.v_peer > 100.0)
        med = [np.median(100.0 * s.p_def[mk & (s.g >= a) & (s.g < b), k - 1]) if np.any(mk & (s.g >= a) & (s.g < b)) else np.nan for a, b in zip(g_edges[:-1], g_edges[1:])]
        ax_g.plot(g_mid, med, color=col, ls=ls, marker="o", markersize=3.2, lw=1.3, label=table.iloc[k - 1]["label"].split(": ", 1)[1])
    ax_g.axvspan(*[100, 250], color=PURPLE, alpha=0.08, label="Low-G band used for PID")
    ax_g.set_xlabel("Measured POA irradiance [W/m2]")
    ax_g.set_ylabel("Power deficit 1 - P_k / P_peer [%]")
    ax_g.set_title("Power deficit vs irradiance, Aug-Sep", fontsize=9.5)
    ax_g.legend(loc="upper right", fontsize=6.8)
    save(fig, "dv_vs_current_signature.png")

    # 3. Daily timeline (showcase) --------------------------------------------------------------
    daily = daily_tracking(s)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 5.4), sharex=True)
    style(ax1)
    style(ax2)
    for k, col, lw in [(1, INK, 1.0), (7, BLUE, 1.2), (2, NAVY, 1.2), (8, PURPLE, 1.2), (3, ORANGE, 1.2), (4, RED, 1.0)]:
        ax1.plot(daily["doy"], daily[f"dv_med_mppt_{k}"], color=col, lw=lw, label=table.iloc[k - 1]["label"])
    for k, col, ls in [(1, INK, "-"), (2, NAVY, "--"), (6, GREEN, "-"), (4, RED, ":")]:
        ax2.plot(daily["doy"], daily[f"i_norm_mppt_{k}"], color=col, ls=ls, lw=1.1, label=table.iloc[k - 1]["label"])
    ax1.set_ylabel("Daily median V_peer - V_k [V]")
    ax1.set_title("Showcase inverter: daily medians against the peer reference (G > 150 W/m2)", fontsize=9.5)
    ax1.legend(loc="upper left", fontsize=7.0, ncol=2)
    ax2.set_ylabel("Daily median I_k / I_peer [-]")
    ax2.set_xlabel("Day of year (2025)")
    ax2.legend(loc="lower left", fontsize=7.0, ncol=2)
    ax2.set_xlim(1, 365)
    save(fig, "mppt_fleet_timeline.png")

    # 4. Held-out verification --------------------------------------------------------------------
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for ax in (ax_a, ax_b, ax_c):
        style(ax)
    rs = heldout[(heldout["true_class"] == "High Series Resistance R_s") & (heldout["predicted_class"] == "High Series Resistance R_s")]
    top = max(4.5, float(rs["rs_true_ohm"].max()) * 1.1) if len(rs) else 4.5
    ax_a.plot([0, top], [0, top], color=GREY, ls="--", lw=1.0, label="1:1")
    ax_a.scatter(rs["rs_true_ohm"], rs["rs_est_ohm"], s=14, alpha=0.6, color=ORANGE, label="Held-out R_s windows")
    ax_a.set_xlim(0, top)
    ax_a.set_ylim(0, top)
    ax_a.set_xlabel("True extra R_s, window mean [Ohm]")
    ax_a.set_ylabel("Estimated slope dV/dI [Ohm]")
    ax_a.set_title("Series resistance recovery", fontsize=9.5)
    ax_a.legend(loc="upper left", fontsize=7.5)

    soil = heldout[(heldout["true_class"] == "Uniform Soiling") & (heldout["predicted_class"] == "Uniform Soiling")]
    top = max(6.0, float(soil["soiling_true_pct"].max()) * 1.1) if len(soil) else 6.0
    ax_b.plot([0, top], [0, top], color=GREY, ls="--", lw=1.0, label="1:1")
    ax_b.scatter(soil["soiling_true_pct"], soil["soiling_est_pct"], s=14, alpha=0.6, color=GREEN, label="Held-out soiling windows")
    ax_b.set_xlim(0, top)
    ax_b.set_ylim(0, top)
    ax_b.set_xlabel("True soiling, window mean [%]")
    ax_b.set_ylabel("Estimated current deficit [%]")
    ax_b.set_title("Soiling recovery", fontsize=9.5)
    ax_b.legend(loc="upper left", fontsize=7.5)

    x = np.arange(len(FAULT_CLASSES))
    ax_c.bar(x - 0.18, [ver["tuning"]["class_metrics"][c]["f1"] for c in FAULT_CLASSES], width=0.36, color=GREY, label="Tuning (in-sample)")
    ax_c.bar(x + 0.18, [ver["heldout"]["class_metrics"][c]["f1"] for c in FAULT_CLASSES], width=0.36, color=BLUE, label="Held-out")
    ax_c.set_xticks(x, [SHORT[c].replace(" ", "\n") for c in FAULT_CLASSES], fontsize=7.5)
    ax_c.set_ylim(0, 1.12)
    ax_c.set_ylabel("F1 [-]")
    ax_c.set_title(f"Window F1, held-out macro-F1 = {ver['heldout']['macro_f1']:.3f}", fontsize=9.5)
    ax_c.legend(loc="upper right", fontsize=7.0)
    save(fig, "verification_recovery.png")


def _lab_payload(spec: InverterStringSpec, soiling_pct: float) -> dict:
    """Model curves for the interactive page; the browser only combines them, it does not re-model."""
    g_grid = list(range(200, 1001, 100))
    t_amb = 22.0
    healthy, presets = [], {"PID": [], "Shading": [], "Soiling": []}
    states = {
        "PID": SubstringFaultState(n_pid_modules=4, pid_rsh_module_ohm=35.0),
        "Shading": SubstringFaultState(n_shaded_substrings=6, shaded_transmission_frac=0.15),
        "Soiling": SubstringFaultState(soiling_loss_frac=soiling_pct / 100.0),
    }
    for g in g_grid:
        tc = float(cell_temperature_c(g, t_amb, spec))
        r = sweep_string_iv(g, tc, SubstringFaultState(), spec=spec, n_points=400)
        # Healthy curve as per-sub-string voltage: shorted diodes and R_s combine with it exactly
        healthy.append({"i_a": np.round(r["i_a"], 4).tolist(), "v_sub_v": np.round(r["v_v"] / spec.n_substrings, 5).tolist()})
        for name, st in states.items():
            p = sweep_string_iv(g, tc, st, spec=spec, n_points=200)
            presets[name].append({
                "v_v": np.round(p["v_v"], 2).tolist(), "i_a": np.round(p["i_a"], 4).tolist(),
                "v_mppt_v": round(p["v_mppt_v"], 2), "i_mppt_a": round(p["i_mppt_a"], 4),
            })
    return {"g_grid": g_grid, "t_amb_c": t_amb, "n_substrings": spec.n_substrings, "healthy": healthy, "presets": presets}


def build_dashboard_payload(sc: dict, ver: dict, spec: InverterStringSpec) -> dict:
    """Compact JSON payload for docs/data.js powering the interactive page."""
    s, table = sc["samples"], sc["table"]
    d0, d1 = SHOWCASE_DAYS
    m = np.flatnonzero((s.doy >= d0) & (s.doy <= d1) & (s.g > 110.0) & (s.v_peer > 100.0))[::4]
    daily = daily_tracking(s)
    dates = (pd.Timestamp("2025-01-01") + pd.to_timedelta(daily["doy"] - 1, unit="D")).dt.strftime("%Y-%m-%d").tolist()
    channels = {}
    for k in range(1, spec.n_mppt + 1):
        row = table.iloc[k - 1]
        channels[row["channel"]] = {
            "label": row["label"],
            "fault_class": row["injected_fault"],
            "scatter": {
                "i_a": np.round(s.i[m, k - 1], 2).tolist(),
                "dv_v": np.round(s.v_peer[m] - s.v[m, k - 1], 2).tolist(),
                "g_wm2": np.round(s.g[m], 0).tolist(),
                "p_def_pct": np.round(100.0 * s.p_def[m, k - 1], 3).tolist(),
            },
            "fit": {"intercept_v": round(float(row["dv_intercept_v"]), 3), "slope_ohm": round(float(row["dv_slope_ohm"]), 4)},
            "daily": {
                "date": dates,
                "i_norm": np.round(daily[f"i_norm_mppt_{k}"], 4).tolist(),
                "dv_med": np.round(daily[f"dv_med_mppt_{k}"], 2).tolist(),
            },
        }
    cols = [
        "channel", "label", "injected_fault", "predicted_class", "n_bd_true", "n_bd_est", "rs_true_ohm", "rs_est_ohm",
        "soiling_true_pct", "soiling_est_pct", "p_def_lo", "annual_energy_mwh", "lost_energy_true_mwh", "lost_energy_est_mwh",
    ]
    fleet = table[cols].copy()
    fleet["p_def_lo"] = 100.0 * fleet["p_def_lo"]
    soil_row = table[table["injected_fault"] == "Uniform Soiling"]
    soil_pct = float(soil_row["soiling_true_pct"].iloc[0]) if len(soil_row) else 3.0
    return {
        "spec": {
            "n_mppt": spec.n_mppt,
            "n_substrings": spec.n_substrings,
            "v_mp_string_stc_v": round(spec.v_mp_string_stc_v, 1),
            "i_mp_stc_a": spec.i_mp_stc_a,
        },
        "showcase_days": list(SHOWCASE_DAYS),
        "verification": ver,
        "fleet_table": json.loads(fleet.round(4).to_json(orient="records")),
        "channels": channels,
        "lab": _lab_payload(spec, round(soil_pct, 1)),
    }


def run_full_pipeline(
    output_root: Path | None = None,
    tuning_seeds: tuple[int, ...] = TUNING_SEEDS,
    heldout_seeds: tuple[int, ...] = HELDOUT_SEEDS,
    workers: int | None = None,
    figures: bool = True,
) -> dict:
    """Tune, evaluate on held-out inverters, run the showcase and write results/ and docs/."""
    root = output_root or REPO_ROOT
    results_dir = root / "results"
    docs_dir = root / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    if set(tuning_seeds) & set(heldout_seeds) or SHOWCASE_SEED in tuning_seeds:
        raise ValueError("tuning seeds must not overlap the held-out or showcase seeds")
    spec = InverterStringSpec()
    n_workers = workers if workers is not None else min(os.cpu_count() or 1, len(tuning_seeds) + len(heldout_seeds))

    runs = _run_seeds(tuple(tuning_seeds) + tuple(heldout_seeds), n_workers)
    tuning_runs, heldout_runs = runs[: len(tuning_seeds)], runs[len(tuning_seeds) :]
    tuning = pd.concat([w for w, _ in tuning_runs], ignore_index=True)
    heldout = pd.concat([w for w, _ in heldout_runs], ignore_index=True)

    # Thresholds see only the fault-free windows of the tuning inverters
    thresholds = fit_thresholds(tuning[tuning["true_class"] == "Healthy"], THRESHOLD_QUANTILE)
    tuning_pred = classify(tuning, thresholds, spec)
    heldout_pred = classify(heldout, thresholds, spec)
    sc = _showcase(spec, thresholds)

    ver = {
        "tuning_seeds": list(tuning_seeds),
        "heldout_seeds": list(heldout_seeds),
        "showcase_seed": SHOWCASE_SEED,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "min_impact_pct": MIN_IMPACT_PCT,
        "thresholds": {k: round(v, 6) for k, v in thresholds.items()},
        "tuning": _split_report(tuning_pred, {seed: scen for (_, scen), seed in zip(tuning_runs, tuning_seeds)}),
        "heldout": _split_report(heldout_pred, {seed: scen for (_, scen), seed in zip(heldout_runs, heldout_seeds)}),
        "showcase": sc["report"],
    }
    ver = json.loads(json.dumps(ver))  # same object as written to verification.json (string keys)

    table_cols = [
        "channel", "label", "injected_fault", "true_class", "predicted_class", "n_bd_true", "n_bd_est", "rs_true_ohm", "rs_est_ohm",
        "soiling_true_pct", "soiling_est_pct", "pid_rsh_true_ohm", "dv_slope_ohm", "dv_intercept_v", "v_def_hi", "i_def_hi",
        "p_def_hi", "p_def_lo", "pid_droop", "shade_score", "shade_az_deg", "shade_el_deg", "true_loss_pct",
        "annual_energy_mwh", "lost_energy_true_mwh", "lost_energy_est_mwh",
    ]
    sc["table"][table_cols].round(4).to_csv(results_dir / "mppt_channel_diagnostics.csv", index=False)
    daily_tracking(sc["samples"]).round(5).to_csv(results_dir / "daily_fleet_tracking.csv", index=False)
    window_cols = [
        "seed", "channel", "window", "day_from", "day_to", "true_class", "predicted_class", "true_loss_pct",
        "n_bd_true", "n_bd_est", "rs_true_ohm", "rs_est_ohm", "soiling_true_pct", "soiling_est_pct",
        *[feat for _, feat in DETECTORS],
    ]
    heldout_pred[window_cols].round(5).to_csv(results_dir / "heldout_window_predictions.csv", index=False)
    (results_dir / "verification.json").write_text(json.dumps(ver, indent=2), encoding="utf-8")
    (results_dir / "summary.md").write_text(summary_markdown(ver, sc["table"]), encoding="utf-8")

    if figures:
        generate_plots(sc, ver, heldout_pred, spec, docs_dir / "img")
        payload = build_dashboard_payload(sc, ver, spec)
        (docs_dir / ".nojekyll").write_text("", encoding="utf-8")
        (docs_dir / "data.js").write_text("window.DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n", encoding="utf-8")

    return {"verification": ver, "showcase_table": sc["table"], "heldout": heldout_pred, "tuning": tuning_pred}
