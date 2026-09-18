import json
import subprocess
import sys

import pytest

from cardd.metrics_schema import validate_metrics_schema
from cardd.promote import (
    MODEL_CARD_END_MARKER,
    MODEL_CARD_START_MARKER,
    README_END_MARKER,
    README_START_MARKER,
    SNAPSHOT_MAP50_95_END_MARKER,
    SNAPSHOT_MAP50_95_START_MARKER,
    SNAPSHOT_MAP50_END_MARKER,
    SNAPSHOT_MAP50_START_MARKER,
    SNAPSHOT_RECALL_END_MARKER,
    SNAPSHOT_RECALL_START_MARKER,
    apply_promotion,
    build_champion_pointer,
    decide_promotion,
    splice_inline_marker,
    splice_marked_section,
)

# Every README fixture below needs the three inline snapshot markers too, on top of the block
# README_START/END_MARKER pair - apply_promotion() now spliced both, and fails loudly (like the
# block splice) if a marker is missing. Kept on one line each, matching how they actually sit
# inside a real markdown table cell (see promote.py's SNAPSHOT_* comments for why - a marker on
# its own line would break the table).
SNAPSHOT_ROW = (
    f"| recall | {SNAPSHOT_RECALL_START_MARKER}old{SNAPSHOT_RECALL_END_MARKER} |\n"
    f"| mAP50 | {SNAPSHOT_MAP50_START_MARKER}old{SNAPSHOT_MAP50_END_MARKER} |\n"
    f"| mAP50-95 | {SNAPSHOT_MAP50_95_START_MARKER}old{SNAPSHOT_MAP50_95_END_MARKER} |\n"
)


def make_metrics(recall, map50_95=0.5, run_name="run", split="test"):
    return {
        "schema_version": 1,
        "run_name": run_name,
        "config_path": "configs/experiments/01_baseline.yaml",
        "git_sha": "abc1234",
        "mlflow_run_id": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "train_seconds": 100.0,
        "weights_path": "runs/run/weights/best.pt",
        "class_names": ["dent", "scratch"],
        "splits": {
            split: {
                "images": 10,
                "instances": 20,
                "aggregate": {
                    "precision": 0.7, "recall": recall, "map50": 0.6, "map50_95": map50_95,
                },
                "per_class": [
                    {"class_id": 0, "class_name": "dent", "precision": 0.7, "recall": recall,
                     "f1": 0.7, "ap50": 0.6, "ap50_95": map50_95},
                ],
            }
        },
    }


def test_decide_promotion_promotes_on_recall_improvement():
    champion = make_metrics(recall=0.60)
    challenger = make_metrics(recall=0.65)
    decision = decide_promotion(champion, challenger)
    assert decision["promote"] is True
    assert decision["delta_recall"] == pytest.approx(0.05)


def test_decide_promotion_rejects_on_recall_regression():
    champion = make_metrics(recall=0.60)
    challenger = make_metrics(recall=0.55)
    decision = decide_promotion(champion, challenger)
    assert decision["promote"] is False
    assert decision["delta_recall"] == pytest.approx(-0.05)


def test_decide_promotion_respects_tolerance():
    champion = make_metrics(recall=0.60)
    challenger = make_metrics(recall=0.58)  # -0.02, within a 0.03 tolerance
    assert decide_promotion(champion, challenger, recall_tolerance=0.03)["promote"] is True
    assert decide_promotion(champion, challenger, recall_tolerance=0.01)["promote"] is False


def test_decide_promotion_raises_on_missing_split():
    champion = make_metrics(recall=0.60, split="test")
    challenger = make_metrics(recall=0.65, split="val")
    with pytest.raises(ValueError, match="challenger"):
        decide_promotion(champion, challenger, split="test")


def test_build_champion_pointer_shape():
    pointer = build_champion_pointer(
        run_name="03_tuned",
        release_tag="champion-03_tuned-20260101",
        rolling_release_tag="champion",
        weights_url="https://github.com/JShi12/vehicle-damage-detection/releases/download/champion/best.pt",
        git_sha="abc1234",
    )
    assert pointer["run_name"] == "03_tuned"
    assert pointer["release_tag"] == "champion-03_tuned-20260101"
    assert pointer["rolling_release_tag"] == "champion"
    assert pointer["metrics_path"] == "models/champion_metrics.json"
    assert "promoted_at" in pointer


def test_splice_marked_section_replaces_between_markers():
    readme = (
        "# Title\n\n## Current champion\n\n"
        f"{README_START_MARKER}\nold content\n{README_END_MARKER}\n\n"
        "## Next section\n"
    )
    updated = splice_marked_section(readme, "new content", README_START_MARKER, README_END_MARKER)
    assert "old content" not in updated
    assert "new content" in updated
    assert "## Next section" in updated
    assert README_START_MARKER in updated
    assert README_END_MARKER in updated


def test_splice_marked_section_raises_if_markers_missing():
    with pytest.raises(ValueError, match="markers"):
        splice_marked_section(
            "# Title\n\nno markers here\n", "new content", README_START_MARKER, README_END_MARKER
        )


def test_splice_marked_section_marker_pairs_are_independent():
    # A document with only the model-card markers must not be touched by a splice targeting the
    # README markers - confirms the two marker pairs can never collide.
    text = f"{MODEL_CARD_START_MARKER}\nsome content\n{MODEL_CARD_END_MARKER}\n"
    with pytest.raises(ValueError, match="markers"):
        splice_marked_section(text, "new content", README_START_MARKER, README_END_MARKER)


def test_splice_inline_marker_replaces_between_markers_on_one_line():
    cell = f"| recall | {SNAPSHOT_RECALL_START_MARKER}old{SNAPSHOT_RECALL_END_MARKER} |"
    updated = splice_inline_marker(
        cell, "**0.708**", SNAPSHOT_RECALL_START_MARKER, SNAPSHOT_RECALL_END_MARKER
    )
    expected = f"| recall | {SNAPSHOT_RECALL_START_MARKER}**0.708**{SNAPSHOT_RECALL_END_MARKER} |"
    assert updated == expected
    # crucially, no newline was introduced - the row must stay a single line or the table breaks
    assert "\n" not in updated


def test_splice_inline_marker_raises_if_markers_missing():
    with pytest.raises(ValueError, match="markers"):
        splice_inline_marker(
            "| recall | 0.6 |", "0.7", SNAPSHOT_RECALL_START_MARKER, SNAPSHOT_RECALL_END_MARKER
        )


def test_apply_promotion_writes_all_files(tmp_path):
    challenger_metrics_path = tmp_path / "challenger_metrics.json"
    champion_json_path = tmp_path / "models" / "champion.json"
    champion_metrics_path = tmp_path / "models" / "champion_metrics.json"
    readme_path = tmp_path / "README.md"
    model_card_path = tmp_path / "MODEL_CARD.md"

    challenger = make_metrics(recall=0.7, run_name="03_tuned")
    challenger_metrics_path.write_text(json.dumps(challenger))
    readme_path.write_text(
        "# Title\n\n## Project snapshot\n\n" + SNAPSHOT_ROW +
        "\n## Current champion\n\n"
        f"{README_START_MARKER}\nold\n{README_END_MARKER}\n"
    )
    model_card_path.write_text(
        "# Model Card\n\n## Quantitative Results\n\n"
        f"{MODEL_CARD_START_MARKER}\nold\n{MODEL_CARD_END_MARKER}\n\n"
        "**Analysis**: written for a different run, unrelated to automation.\n"
    )

    pointer = apply_promotion(
        challenger_metrics_path=challenger_metrics_path,
        champion_json_path=champion_json_path,
        champion_metrics_path=champion_metrics_path,
        readme_path=readme_path,
        release_tag="champion-03_tuned-20260101",
        rolling_release_tag="champion",
        weights_url="https://github.com/JShi12/vehicle-damage-detection/releases/download/champion/best.pt",
        git_sha="abc1234",
        model_card_path=model_card_path,
    )

    assert champion_metrics_path.exists()
    written_metrics = json.loads(champion_metrics_path.read_text())
    validate_metrics_schema(written_metrics)
    assert written_metrics["run_name"] == "03_tuned"

    assert champion_json_path.exists()
    assert json.loads(champion_json_path.read_text()) == pointer

    readme_text = readme_path.read_text()
    assert "old" not in readme_text
    assert "03_tuned" in readme_text
    # snapshot table cells got the real test-split numbers, not just "not old"
    assert f"{SNAPSHOT_RECALL_START_MARKER}**0.700**{SNAPSHOT_RECALL_END_MARKER}" in readme_text

    model_card_text = model_card_path.read_text()
    assert "03_tuned" in model_card_text
    # the marker-delimited block updated, but hand-written prose outside it must survive untouched
    assert "written for a different run, unrelated to automation" in model_card_text


def test_apply_promotion_skips_model_card_when_not_given(tmp_path):
    challenger_metrics_path = tmp_path / "challenger_metrics.json"
    readme_path = tmp_path / "README.md"
    challenger_metrics_path.write_text(json.dumps(make_metrics(recall=0.7, run_name="03_tuned")))
    readme_path.write_text(SNAPSHOT_ROW + f"{README_START_MARKER}\nold\n{README_END_MARKER}\n")

    apply_promotion(
        challenger_metrics_path=challenger_metrics_path,
        champion_json_path=tmp_path / "models" / "champion.json",
        champion_metrics_path=tmp_path / "models" / "champion_metrics.json",
        readme_path=readme_path,
        release_tag="champion-03_tuned-20260101",
        rolling_release_tag="champion",
        weights_url="https://example.invalid/best.pt",
        # model_card_path omitted entirely - must not raise or create anything
    )

    assert not (tmp_path / "MODEL_CARD.md").exists()


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "cardd.cli.promote", *args],
        capture_output=True, text=True,
    )


def test_cli_check_only_does_not_write_files(tmp_path):
    champion_metrics = tmp_path / "champion_metrics.json"
    challenger_metrics = tmp_path / "challenger_metrics.json"
    champion_metrics.write_text(json.dumps(make_metrics(recall=0.60)))
    challenger_metrics.write_text(json.dumps(make_metrics(recall=0.70, run_name="challenger")))
    decision_out = tmp_path / "decision.json"
    champion_json = tmp_path / "models" / "champion.json"

    result = _run_cli(
        "--check-only",
        "--champion-metrics", str(champion_metrics),
        "--challenger-metrics", str(challenger_metrics),
        "--champion-json", str(champion_json),
        "--decision-out", str(decision_out),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(decision_out.read_text())["promote"] is True
    assert not champion_json.exists()


def test_cli_apply_writes_files(tmp_path):
    # --no-model-card here deliberately: this test doesn't set up a MODEL_CARD.md fixture, and
    # without an explicit override the CLI's default (--model-card MODEL_CARD.md, relative) would
    # otherwise resolve against the real repo root pytest runs from - see
    # test_cli_apply_writes_model_card below for that path specifically, with its own tmp_path file.
    champion_metrics = tmp_path / "champion_metrics.json"
    challenger_metrics = tmp_path / "challenger_metrics.json"
    champion_metrics.write_text(json.dumps(make_metrics(recall=0.60)))
    challenger_metrics.write_text(json.dumps(make_metrics(recall=0.70, run_name="challenger")))
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\n{SNAPSHOT_ROW}\n{README_START_MARKER}\nold\n{README_END_MARKER}\n"
    )
    champion_json = tmp_path / "models" / "champion.json"

    result = _run_cli(
        "--champion-metrics", str(champion_metrics),
        "--challenger-metrics", str(challenger_metrics),
        "--challenger-weights-url", "https://example.invalid/best.pt",
        "--release-tag", "champion-challenger-20260101",
        "--readme", str(readme),
        "--champion-json", str(champion_json),
        "--no-model-card",
    )

    assert result.returncode == 0, result.stderr
    assert champion_json.exists()
    assert json.loads(champion_metrics.read_text())["run_name"] == "challenger"
    assert "challenger" in readme.read_text()


def test_cli_apply_writes_model_card(tmp_path):
    champion_metrics = tmp_path / "champion_metrics.json"
    challenger_metrics = tmp_path / "challenger_metrics.json"
    champion_metrics.write_text(json.dumps(make_metrics(recall=0.60)))
    challenger_metrics.write_text(json.dumps(make_metrics(recall=0.70, run_name="challenger")))
    readme = tmp_path / "README.md"
    readme.write_text(SNAPSHOT_ROW + f"{README_START_MARKER}\nold\n{README_END_MARKER}\n")
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text(f"{MODEL_CARD_START_MARKER}\nold\n{MODEL_CARD_END_MARKER}\n")
    champion_json = tmp_path / "models" / "champion.json"

    result = _run_cli(
        "--champion-metrics", str(champion_metrics),
        "--challenger-metrics", str(challenger_metrics),
        "--challenger-weights-url", "https://example.invalid/best.pt",
        "--release-tag", "champion-challenger-20260101",
        "--readme", str(readme),
        "--champion-json", str(champion_json),
        "--model-card", str(model_card),
    )

    assert result.returncode == 0, result.stderr
    assert "challenger" in model_card.read_text()
    assert "old" not in model_card.read_text()
