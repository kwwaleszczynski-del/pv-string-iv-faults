"""Tests for pipeline execution, generated figures, results tables, and interactive dashboard files."""

from __future__ import annotations

import json
from pathlib import Path

from pv_iv_faults.pipeline import REPO_ROOT, run_full_pipeline


def test_pipeline_writes_results_and_docs():
    out = run_full_pipeline(output_root=REPO_ROOT)
    assert "verification" in out
    assert (REPO_ROOT / "results" / "mppt_channel_diagnostics.csv").is_file()
    assert (REPO_ROOT / "results" / "daily_fleet_tracking.csv").is_file()
    assert (REPO_ROOT / "results" / "verification.json").is_file()


def test_four_matplotlib_figures_exist_and_non_empty():
    img_dir = REPO_ROOT / "docs" / "img"
    expected = [
        "iv_pv_curves.png",
        "dv_vs_current_signature.png",
        "mppt_fleet_timeline.png",
        "verification_recovery.png",
    ]
    for name in expected:
        p = img_dir / name
        assert p.is_file(), f"Missing plot {name}"
        assert p.stat().st_size > 15_000, f"Plot {name} is unexpectedly small"


def test_dashboard_data_js_and_nojekyll():
    docs_dir = REPO_ROOT / "docs"
    assert (docs_dir / ".nojekyll").is_file()
    data_js = (docs_dir / "data.js").read_text(encoding="utf-8")
    assert data_js.startswith("window.DATA = ")
    payload = json.loads(data_js.removeprefix("window.DATA = ").strip().rstrip(";"))
    assert payload["spec"]["n_substrings"] == 72
    assert len(payload["fleet_table"]) == 12
    assert "MPPT 1" in payload["channels"]
    assert "MPPT 6" in payload["channels"]


def test_dashboard_index_html_contains_required_controls():
    html = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    for required in [
        "plotly.js-dist-min@2.35.2",
        "MPPT 1: Healthy",
        "MPPT 2: 2 Shorted Bypass Diodes",
        "MPPT 3: Corroded MC4 (+2.5 Ohm R_s)",
        "MPPT 4: PID Shunt Degradation",
        "MPPT 5: Morning Row/Chimney Shading",
        "MPPT 6: 4.5 % Soiling",
        "slider-nbd",
        "slider-rs",
        "slider-g",
    ]:
        assert required in html
