"""Real-artifact checks opt in with STEP3_ARTIFACT_DIR, independently of implementation imports."""

import json

import pytest

from helpers.step3_spec import (CLASSES, assert_artifact_schema, assert_number, assert_prediction_row,
                               assert_run_schema, read_json, resolve_recorded_path, sha256)

pytestmark = pytest.mark.artifacts


def test_real_artifact_manifest_schema_and_file_hashes(step3_artifact_dir):
    # Protect plan §5 Bước 3/§8/§11 provenance and contract artifact schema/invariant 6.
    root = step3_artifact_dir
    artifact = read_json(root / "artifact.json")
    assert_artifact_schema(artifact)
    manifest = resolve_recorded_path(artifact["run_manifest"]["path"], root)
    assert sha256(manifest) == artifact["run_manifest"]["sha256"]
    run = read_json(manifest)
    assert_run_schema(run)
    for kind in ("best", "last"):
        actual = sha256(root / f"{kind}.pt")
        assert actual == artifact[f"checkpoint_{kind}_sha256"] == run["checkpoints"][f"{kind}_sha256"]
    assert artifact["smoke"] is False and run["smoke"] is False
    assert run["git_dirty"] is False
    for key in ("model_id", "seed", "config_hash", "view_signature"):
        assert artifact[key] == run[key]
    assert artifact["created_by_git_commit"] == run["git_commit"]


def test_real_artifact_calibration_schema(step3_artifact_dir):
    # Protect plan §5 Bước 3/§7.3 and contract calibration report: AP separate from F1-optimal P/R.
    report = read_json(step3_artifact_dir / "calibration_per_class.json")
    assert {"split", "n_images", "overall", "per_class", "source", "ultralytics_version"} <= report.keys()
    assert report["split"] == "calibration" and report["n_images"] == 460
    assert type(report["n_images"]) is int
    assert isinstance(report["source"], str) and report["source"]
    assert isinstance(report["ultralytics_version"], str) and report["ultralytics_version"]
    # Contract calls this "infer params" without prescribing its spelling.
    params = [report[key] for key in ("infer_params", "inference_params", "infer") if key in report]
    assert params, "Calibration report must disclose its inference parameters"
    assert all(value == read_json(step3_artifact_dir / "artifact.json")["inference_params"] for value in params)
    overall = report["overall"]
    assert {"map50", "map50_95", "precision", "recall", "pr_definition", "pr_conf"} <= overall.keys()
    assert isinstance(overall["pr_definition"], str) and overall["pr_definition"]
    for key in ("map50", "map50_95", "precision", "recall", "pr_conf"):
        assert_number(overall[key])
        assert overall[key] <= 1
    assert set(report["per_class"]) == set(CLASSES)
    for metrics in report["per_class"].values():
        assert {"n_gt_boxes", "precision", "recall", "ap50", "ap50_95"} <= metrics.keys()
        assert type(metrics["n_gt_boxes"]) is int and metrics["n_gt_boxes"] >= 0
        for key in ("precision", "recall", "ap50", "ap50_95"):
            assert_number(metrics[key])
            assert metrics[key] <= 1


@pytest.mark.dataset
def test_real_artifact_prediction_rows(step3_artifact_dir, real_dataset_root):
    # Protect plan §1.2/§6/§8 and contract invariant 5: full 460/306 images, no silent errors.
    from pcb_lab.data.samples import ManifestDataset
    root = step3_artifact_dir
    meta = read_json(root / "artifact.json")
    run = read_json(resolve_recorded_path(meta["run_manifest"]["path"], root))
    for part, count in (("calibration", 460), ("fusion", 306)):
        samples = sorted(ManifestDataset(real_dataset_root, part), key=lambda s: s.sample_id)
        rows = [json.loads(line) for line in (root / f"preds_{part}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == len(samples) == count
        assert [row["sample_id"] for row in rows] == [sample.sample_id for sample in samples]
        assert len({row["sample_id"] for row in rows}) == count
        for row, sample in zip(rows, samples):
            assert_prediction_row(row, sample, meta, run)
            assert row["error_reason"] is None, (row["sample_id"], row["error_reason"])
            assert len(row["detections"]) < meta["inference_params"]["max_det"], "Investigate max_det saturation"
