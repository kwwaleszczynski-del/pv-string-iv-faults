"""Evaluation rules, an end-to-end run on a reduced split, and agreement of the committed outputs."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pv_iv_faults.diagnostics import FAULT_CLASSES, NO_DECISION
from pv_iv_faults.evaluation import MIN_IMPACT_PCT, classification_scores, window_label
from pv_iv_faults.pipeline import REPO_ROOT, run_full_pipeline
from pv_iv_faults.simulator import ChannelScenario

RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"


def _data_js(path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("window.DATA = ")
    return json.loads(text.removeprefix("window.DATA = ").strip().rstrip(";"))


def test_window_labels():
    sc = ChannelScenario("MPPT 1", "Shorted Bypass Diode", onset_day=100, n_bd=1)
    assert window_label(sc, 1, 14, 0.0) == "Healthy"
    assert window_label(sc, 99, 112, 1.0) == "Transition"
    assert window_label(sc, 113, 126, MIN_IMPACT_PCT) == "Shorted Bypass Diode"
    assert window_label(sc, 113, 126, MIN_IMPACT_PCT / 2) == "Below impact"
    assert window_label(ChannelScenario("MPPT 2"), 113, 126, 0.0) == "Healthy"


def test_scores_count_abstentions_as_misses_only():
    perfect = classification_scores(FAULT_CLASSES, FAULT_CLASSES)
    assert perfect["macro_f1"] == 1.0
    sc = classification_scores(["Healthy", "Uniform Soiling"], ["Healthy", NO_DECISION])
    assert sc["class_metrics"]["Uniform Soiling"]["fn"] == 1
    assert sc["class_metrics"]["Healthy"]["precision"] == 1.0


def test_pipeline_refuses_overlapping_tuning_and_heldout_seeds(tmp_path):
    with pytest.raises(ValueError):
        run_full_pipeline(output_root=tmp_path, tuning_seeds=(1, 2), heldout_seeds=(2, 3), workers=1)


def test_reduced_pipeline_runs_end_to_end(tmp_path):
    out = run_full_pipeline(output_root=tmp_path, tuning_seeds=(1001,), heldout_seeds=(2001,), workers=1)
    ver = out["verification"]
    assert ver["tuning_seeds"] == [1001] and ver["heldout_seeds"] == [2001]
    assert set(ver["heldout"]["class_metrics"]) == set(FAULT_CLASSES)
    for name in ("mppt_channel_diagnostics.csv", "daily_fleet_tracking.csv", "heldout_window_predictions.csv", "verification.json", "summary.md"):
        assert (tmp_path / "results" / name).is_file()
    for name in ("iv_pv_curves.png", "dv_vs_current_signature.png", "mppt_fleet_timeline.png", "verification_recovery.png"):
        assert (tmp_path / "docs" / "img" / name).stat().st_size > 15_000
    held = pd.read_csv(tmp_path / "results" / "heldout_window_predictions.csv")
    assert set(held["seed"]) == {2001}
    assert _data_js(tmp_path / "docs" / "data.js")["verification"] == ver


def test_committed_tuning_heldout_and_showcase_seeds_are_disjoint():
    ver = json.loads((RESULTS / "verification.json").read_text(encoding="utf-8"))
    tuning, heldout = set(ver["tuning_seeds"]), set(ver["heldout_seeds"])
    assert tuning and heldout and not tuning & heldout
    assert ver["showcase_seed"] not in tuning


def test_readme_quotes_the_generated_summary_verbatim():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for line in (RESULTS / "summary.md").read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert line in readme, f"README is missing generated line: {line[:90]}"


def test_dashboard_data_matches_results():
    ver = json.loads((RESULTS / "verification.json").read_text(encoding="utf-8"))
    payload = _data_js(DOCS / "data.js")
    assert payload["verification"] == ver
    table = pd.read_csv(RESULTS / "mppt_channel_diagnostics.csv")
    assert [r["predicted_class"] for r in payload["fleet_table"]] == table["predicted_class"].tolist()
    assert [r["lost_energy_est_mwh"] for r in payload["fleet_table"]] == pytest.approx(table["lost_energy_est_mwh"].tolist())
    assert (DOCS / ".nojekyll").is_file()


def test_dashboard_page_reads_numbers_from_data_only():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    for required in ("plotly.js-dist-min@2.35.2", 'src="data.js"', "slider-nbd", "slider-rs", "slider-g", "channel-select", "conf-table"):
        assert required in html
    for literal in ("14.3", "0.989", "0.968"):
        assert literal not in html
