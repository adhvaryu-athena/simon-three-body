"""
prepare_encounter_nn_mentor_package.py

Create a compact mentor-review folder for the EncounterNN project.

Default source:
    C:\\Aarush\\Physics\\training\\encounter_nn

Default destination:
    C:\\Aarush\\Physics\\training\\encounter_nn\\send to mentor

What it does:
    - Copies only important code, summaries, selected final outputs, final model files, and key datasets.
    - Skips smoke tests, raw shard folders, caches, plots, and unnecessary intermediate clutter.
    - Writes README_FOR_MENTOR.txt explaining what is included.
    - Writes MANIFEST.csv listing copied/missing files.

Run:
    python -B prepare_encounter_nn_mentor_package.py

Optional:
    python -B prepare_encounter_nn_mentor_package.py --dry-run
    python -B prepare_encounter_nn_mentor_package.py --include-large
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_SRC = Path(r"C:\Aarush\Physics\training\encounter_nn")
DEFAULT_DST = DEFAULT_SRC / "send to mentor"

# Optional: scalar Zone-3 model lives in the older encounter_training folder in your setup.
DEFAULT_SCALAR_MODEL = Path(r"C:\Aarush\Physics\training\encounter_training\pair_correction_nn_v4_bounded.pt")

# Files/folders matching these phrases will be excluded unless explicitly listed.
EXCLUDE_NAME_PARTS = [
    "__pycache__",
    ".ipynb_checkpoints",
    "smoke",
    "Smoke",
    "SMOKE",
]

# Main code files from the EncounterNN journey.
CODE_FILES = [
    # Data generation and merge/inspection
    "generate_encounter_surrogate_data_v1.py",
    "generate_encounter_surrogate_data_v3_rollout_local.py",
    "merge_encounter_surrogate_data_v1.py",
    "merge_encounter_surrogate_data_v2.py",
    "inspect_encounter_surrogate_data_v1.py",
    "inspect_encounter_surrogate_data_v2.py",

    # Training variants tried
    "train_encounter_surrogate_v1.py",
    "train_encounter_surrogate_v2_velocitysafe.py",
    "train_encounter_surrogate_v2_velocitysafe_fixed.py",
    "train_encounter_surrogate_v2_velocitysafe_fixed2.py",
    "train_encounter_surrogate_v3_compact.py",
    "train_encounter_surrogate_v4_hybrid.py",

    # Evaluation / diagnostics
    "evaluate_gated_encounter_surrogate_rollout_v1.py",
    "diagnose_single_event_replay_v1.py",
    "diagnose_event_nearest_neighbors_v1.py",
    "diagnose_single_event_replay_knn_v1.py",
    "diagnose_rollout_error_timeline_v1.py",
    "diagnose_rollout_error_timeline_v1b.py",
    "diagnose_rollout_error_timeline_v1c.py",
    "knn_residual_diag_v1.py",

    # Final ablation evaluator
    "pair_eval_encounterNN_ablation_v1.py",
]

# Important one-off top-level results. Missing files are okay; the script records them.
TOP_LEVEL_RESULT_FILES = [
    # Major training summaries
    "encounter_surrogate_v1_large1_summary.txt",
    "encounter_surrogate_v2_velocitysafe_large1_summary.txt",
    "encounter_surrogate_v2_event_enriched_velocitysafe_summary.txt",
    "encounter_surrogate_v3_compact_event_enriched_summary.txt",
    "encounter_surrogate_v4_hybrid_event_enriched_summary.txt",
    "encounter_surrogate_v3_rollout_local_velocitysafe_summary.txt",

    # Gate / sweep summaries
    "encounter_surrogate_v1_gated_eval.txt",
    "encounter_surrogate_v2_velocitysafe_gate_sweep.txt",
    "encounter_surrogate_v2_velocitysafe_fixed_gate_vr_m0p40.txt",
    "encounter_surrogate_v2_event_enriched_gate_sweep.txt",
    "encounter_surrogate_v2_event_enriched_fixed_gate_vr_m0p40.txt",

    # User-created merged/inspection notes, if present
    "new data results.docx",
    "new data results 1.docx",
    "new data results1_local.docx",
    "latest new data results.docx",

    # Final mentor report, if you copied it into this folder
    "EncounterNN_Mentor_Report.pdf",
    "encounter_nn_mentor_report.docx",
]

# Key folders to copy selectively. Only selected file types inside are copied.
KEY_RESULT_FOLDERS = [
    "single_event_replay_v3_rollout_local_t82p56",
    "rollout_surrogate_v2_event_enriched_T100_vr_m0p40",
    "rollout_surrogate_v3_rollout_local_T100_vr_m0p40",
    "rollout_error_timeline_v1c",
    "event_nn_diag_t82p56",
    "knn_residual_diag_v1",
    "single_event_replay_knn_v1",
    "encounterNN_ablation_v1_out",
]

# Within result folders, keep compact text/csv/docx/pdf summaries and metrics.
KEEP_EXTENSIONS_IN_RESULT_FOLDERS = {".txt", ".csv", ".docx", ".pdf", ".json"}
KEEP_NAME_PATTERNS_IN_RESULT_FOLDERS = [
    "*summary*",
    "*metrics*",
    "*sweep*",
    "*events*",
    "*nearest*",
    "*manifest*",
]

# Final files needed to reproduce the final ablation. These can be large.
FINAL_MODEL_AND_DATA = [
    "encounter_surrogate_v3_rollout_local_velocitysafe.pt",
    "encounter_surrogate_v3_rollout_local_enriched.npz",
]

# Intermediate models worth retaining only if --include-large is used.
INTERMEDIATE_MODELS_AND_DATA = [
    "encounter_surrogate_v1_large1.npz",
    "encounter_surrogate_v2_velocitysafe_large1.pt",
    "encounter_surrogate_v2_event_enriched_velocitysafe.pt",
    "encounter_surrogate_v3_compact_event_enriched.pt",
    "encounter_surrogate_v4_hybrid_event_enriched.pt",
]


def should_exclude(path: Path) -> bool:
    s = str(path)
    return any(part in s for part in EXCLUDE_NAME_PARTS)


def copy_file(src: Path, dst: Path, manifest: List[dict], category: str, dry_run: bool) -> None:
    if should_exclude(src):
        manifest.append(row(category, src, dst, "SKIPPED_EXCLUDED", ""))
        return
    if not src.exists():
        manifest.append(row(category, src, dst, "MISSING", ""))
        return
    if src.is_dir():
        manifest.append(row(category, src, dst, "SKIPPED_IS_DIR", ""))
        return

    size = src.stat().st_size
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    manifest.append(row(category, src, dst, "COPIED", size))


def row(category: str, src: Path, dst: Path, status: str, size_bytes) -> dict:
    return {
        "category": category,
        "status": status,
        "size_bytes": size_bytes,
        "source": str(src),
        "destination": str(dst),
    }


def copy_named_files(src_root: Path, dst_root: Path, names: Iterable[str], subdir: str, category: str, manifest: List[dict], dry_run: bool) -> None:
    for name in names:
        src = src_root / name
        dst = dst_root / subdir / name
        copy_file(src, dst, manifest, category, dry_run)


def copy_external_file(src: Path, dst_root: Path, subdir: str, category: str, manifest: List[dict], dry_run: bool) -> None:
    dst = dst_root / subdir / src.name
    copy_file(src, dst, manifest, category, dry_run)


def keep_result_file(path: Path) -> bool:
    if path.suffix.lower() not in KEEP_EXTENSIONS_IN_RESULT_FOLDERS:
        return False
    lower_name = path.name.lower()
    if any(fnmatch.fnmatch(lower_name, pat.lower()) for pat in KEEP_NAME_PATTERNS_IN_RESULT_FOLDERS):
        return True
    # Keep all text/csv files in explicitly selected result folders because they are usually compact.
    return path.suffix.lower() in {".txt", ".csv", ".json"}


def copy_result_folder(src_root: Path, dst_root: Path, folder_name: str, manifest: List[dict], dry_run: bool) -> None:
    src_folder = src_root / folder_name
    if not src_folder.exists():
        manifest.append(row("result_folder", src_folder, dst_root / "results" / folder_name, "MISSING_FOLDER", ""))
        return
    if not src_folder.is_dir():
        manifest.append(row("result_folder", src_folder, dst_root / "results" / folder_name, "SKIPPED_NOT_DIR", ""))
        return

    copied_any = False
    for p in src_folder.rglob("*"):
        if should_exclude(p) or not p.is_file():
            continue
        if keep_result_file(p):
            rel = p.relative_to(src_folder)
            dst = dst_root / "results" / folder_name / rel
            copy_file(p, dst, manifest, "result_folder_file", dry_run)
            copied_any = True
    if not copied_any:
        manifest.append(row("result_folder", src_folder, dst_root / "results" / folder_name, "NO_SELECTED_FILES", ""))


def write_readme(dst_root: Path, dry_run: bool) -> None:
    text = f"""EncounterNN mentor-review package
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Purpose
-------
This folder is a compact review package for the EncounterNN work. It intentionally excludes
smoke tests, raw shard folders, cache files, and most intermediate clutter.

Main final conclusion
---------------------
The EncounterNN residual model successfully learned a local close-encounter exit correction,
but the final full-pipeline ablation showed that it does not replace the existing scalar Zone-3
SIMON method. It improves some velocity metrics, but it is slower and does not improve the
main position time-averaged RMS metric.

Folder structure
----------------
code/      Important Python scripts used across data generation, training, diagnostics, and final ablation.
models/    Final trained EncounterNN model, plus scalar Zone-3 model if found.
data/      Final merged dataset if --include-large was used or if selected as final data.
results/   Key summaries, CSV metrics, event logs, gate sweeps, and final ablation outputs.
MANIFEST.csv  Full list of copied and missing files.

Recommended reading order
-------------------------
1. EncounterNN_Mentor_Report.pdf, if present.
2. results/encounterNN_ablation_v1_out/encounterNN_ablation_summary.txt
3. results/encounterNN_ablation_v1_out/encounter_events.csv
4. results/single_event_replay_v3_rollout_local_t82p56/single_event_replay_summary.txt
5. results/rollout_error_timeline_v1c/rollout_error_timeline_summary.txt
6. code/pair_eval_encounterNN_ablation_v1.py

Important final files
---------------------
Final trained EncounterNN model:
    encounter_surrogate_v3_rollout_local_velocitysafe.pt

Final merged training data:
    encounter_surrogate_v3_rollout_local_enriched.npz

Final full-pipeline evaluator:
    pair_eval_encounterNN_ablation_v1.py

Final ablation output:
    encounterNN_ablation_v1_out/encounterNN_ablation_summary.txt
    encounterNN_ablation_v1_out/encounter_events.csv
"""
    if not dry_run:
        (dst_root / "README_FOR_MENTOR.txt").write_text(text, encoding="utf-8")


def write_manifest(dst_root: Path, manifest: List[dict], dry_run: bool) -> None:
    if dry_run:
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    out = dst_root / "MANIFEST.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "status", "size_bytes", "source", "destination"])
        writer.writeheader()
        writer.writerows(manifest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="Source encounter_nn folder")
    ap.add_argument("--dst", default=str(DEFAULT_DST), help="Destination mentor package folder")
    ap.add_argument("--scalar-model", default=str(DEFAULT_SCALAR_MODEL), help="Optional scalar Zone-3 model path to include if found")
    ap.add_argument("--include-large", action="store_true", help="Also include final/intermediate large .npz/.pt files")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    manifest: List[dict] = []

    if not src_root.exists():
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    if dst_root.exists() and not args.dry_run:
        # Do not delete automatically; keep old package and overwrite/update selected files.
        pass
    elif not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    # Copy important code and compact results.
    copy_named_files(src_root, dst_root, CODE_FILES, "code", "code", manifest, args.dry_run)
    copy_named_files(src_root, dst_root, TOP_LEVEL_RESULT_FILES, "results", "top_level_result", manifest, args.dry_run)

    # Copy final model always; final dataset only with --include-large because it may be large.
    copy_named_files(src_root, dst_root, ["encounter_surrogate_v3_rollout_local_velocitysafe.pt"], "models", "final_model", manifest, args.dry_run)
    if args.include_large:
        copy_named_files(src_root, dst_root, ["encounter_surrogate_v3_rollout_local_enriched.npz"], "data", "final_dataset", manifest, args.dry_run)
        copy_named_files(src_root, dst_root, INTERMEDIATE_MODELS_AND_DATA, "archive_intermediate_large", "intermediate_large", manifest, args.dry_run)
    else:
        manifest.append(row("final_dataset", src_root / "encounter_surrogate_v3_rollout_local_enriched.npz", dst_root / "data" / "encounter_surrogate_v3_rollout_local_enriched.npz", "SKIPPED_USE_INCLUDE_LARGE", ""))

    # Copy scalar Zone-3 model if available, since final evaluator compares against scalarNN.
    scalar_path = Path(args.scalar_model)
    copy_external_file(scalar_path, dst_root, "models", "scalar_zone3_model_external", manifest, args.dry_run)

    # Copy selected result folders.
    for folder in KEY_RESULT_FOLDERS:
        copy_result_folder(src_root, dst_root, folder, manifest, args.dry_run)

    write_readme(dst_root, args.dry_run)
    write_manifest(dst_root, manifest, args.dry_run)

    copied = sum(1 for r in manifest if r["status"] == "COPIED")
    missing = sum(1 for r in manifest if "MISSING" in r["status"])
    skipped = sum(1 for r in manifest if r["status"].startswith("SKIPPED") or r["status"] == "NO_SELECTED_FILES")

    print("=" * 88)
    print("EncounterNN mentor package preparation")
    print(f"source      : {src_root}")
    print(f"destination : {dst_root}")
    print(f"dry_run     : {args.dry_run}")
    print(f"include_large: {args.include_large}")
    print("-" * 88)
    print(f"copied : {copied}")
    print(f"missing: {missing}")
    print(f"skipped: {skipped}")
    if not args.dry_run:
        print(f"manifest: {dst_root / 'MANIFEST.csv'}")
        print(f"readme  : {dst_root / 'README_FOR_MENTOR.txt'}")
    print("=" * 88)

    if args.dry_run:
        print("\nDRY RUN details:")
        for r in manifest:
            print(f"{r['status']:24s} | {r['category']:24s} | {r['source']}")


if __name__ == "__main__":
    main()
