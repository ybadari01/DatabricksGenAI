import json
import yaml
from pathlib import Path
from typing import Optional


def validate_manifest_alignment(
    course_name: str,
    vocareum_share_name: Optional[str] = None,
    manifest_path: Optional[str | Path] = None,
    medallion_lab: Optional[bool] = False,
) -> bool:
    """
    Validate that config values align with the manifest.

    Checks (standard labs)
    ----------------------
    1. course_name matches a lab 'name' in the manifest labs list.
    2. vocareum_share_name (if provided) matches a dataset 'name' for that lab.

    Checks (medallion_lab=True)
    ----------------------------
    1. Manifest file exists and is valid JSON.
    2. Manifest contains a top-level 'course' field.
    3. 'labs.lti' is set to True.

    Raises ValueError with a descriptive message on any mismatch.
    Returns True on success.
    """
    # --- Resolve manifest path ---
    if manifest_path is None:
        if medallion_lab:
            # Default location for Medallion Labs manifests
            manifest_path = Path(__file__).parents[3] / ".binder" / "manifest.json"
        else:
            manifest_path = Path(__file__).parents[7] / "manifest.yaml"

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # --- Parse manifest (JSON for medallion labs, YAML otherwise) ---
    with open(manifest_path) as f:
        if medallion_lab:
            manifest = json.load(f)
        else:
            manifest = yaml.safe_load(f)

    # --- Medallion Labs validation branch ---
    if medallion_lab:
        return _validate_medallion_manifest(manifest, manifest_path)

    # --- Standard Labs validation ---
    return _validate_standard_manifest(manifest, course_name, vocareum_share_name)


def _validate_medallion_manifest(manifest: dict, manifest_path: Path) -> bool:
    """Validate structure of a Medallion Labs manifest.json."""

    # 1. Top-level 'course' field must be present
    course = manifest.get("course")
    if not course:
        raise ValueError(
            f"Medallion Labs manifest at '{manifest_path}' is missing a top-level 'course' field."
        )
    print(f"  [OK] Medallion Labs manifest found for course: '{course}'")

    # 2. labs.lti must be True
    labs = manifest.get("labs", {})
    if not labs.get("lti"):
        raise ValueError(
            f"Medallion Labs manifest for '{course}' does not have 'labs.lti' set to true.\n"
            f"  Found: {labs.get('lti')!r}"
        )
    print(f"  [OK] 'labs.lti' is True.")

    return True


def _validate_standard_manifest(
    manifest: dict,
    course_name: str,
    vocareum_share_name: Optional[str],
) -> bool:
    """Validate a standard YAML manifest against course_name and optional dataset."""

    # 1. Course name check against top-level 'course' field
    manifest_course = manifest.get("course", "")
    if manifest_course != course_name:
        raise ValueError(
            f"course_name '{course_name}' not found in manifest.\n"
            f"  Manifest course: '{manifest_course}'"
        )
    print(f"  [OK] course_name '{course_name}' found in manifest.")

    # 2. Vocareum dataset check against labs.defaults.config.datasets
    if vocareum_share_name:
        datasets = (
            manifest.get("labs", {})
            .get("defaults", {})
            .get("config", {})
            .get("datasets") or []
        )
        dataset_names = [d.get("name", "") for d in datasets]
        if vocareum_share_name not in dataset_names:
            raise ValueError(
                f"vocareum_share_name '{vocareum_share_name}' not found in manifest datasets for '{course_name}'.\n"
                f"  Available datasets: {dataset_names if dataset_names else '(none configured)'}\n"
                f"  Add a 'datasets' entry to labs.defaults.config.datasets in manifest.yaml."
            )
        print(f"  [OK] vocareum_share_name '{vocareum_share_name}' found in manifest datasets.")

    return True