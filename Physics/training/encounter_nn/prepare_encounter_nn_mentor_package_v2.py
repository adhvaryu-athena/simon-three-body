#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_encounter_nn_mentor_package_v2.py

Creates a SMALL mentor-review package for the EncounterNN work.

What changed from the earlier package script:
  1. NO .pt model files and NO .npz data files are copied.
  2. Only the main code files and main result files are copied.
  3. The README explains exactly what each copied file is.
  4. Superseded/missing experimental files from the earlier manifest are not requested.
  5. The destination folder is cleaned by default, so old copied files do not remain.

Run from:
  C:\Aarush\Physics\training\encounter_nn

Command:
  python -B prepare_encounter_nn_mentor_package_v2.py

Output:
  C:\Aarush\Physics\training\encounter_nn\send to mentor
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Safety: never copy binary model/data files to mentor package.
# ---------------------------------------------------------------------------
EXCLUDED_SUFFIXES = {
    ".pt", ".pth", ".npz", ".npy", ".pkl", ".pickle", ".joblib",
    ".zip", ".7z", ".rar", ".tar", ".gz",
}

DEST_FOLDER_NAME = "send to mentor"


@dataclass(frozen=True)
class SelectedFile:
    group: str
    source_rel: str
    dest_rel: str
    purpose: str
    required: bool = True


# ---------------------------------------------------------------------------
# Small curated file list.
#
# Keep this list intentionally short. It is designed for mentor review, not for
# complete reproducibility with every intermediate file.
# ---------------------------------------------------------------------------
SELECTED_FILES: List[SelectedFile] = [
    # Final / important code files
    SelectedFile(
        group="code",
        source_rel="pair_eval_encounterNN_ablation_v1.py",
        dest_rel="01_code/pair_eval_encounterNN_ablation_v1.py",
        purpose=(
            "Final complete-simulation ablation evaluator. Compares IAS15, "
            "SIMON-noNN, SIMON-scalarNN, and SIMON-encounterNN at dt=0.08."
        ),
    ),
    SelectedFile(
        group="code",
        source_rel="generate_encounter_surrogate_data_v3_rollout_local.py",
        dest_rel="01_code/generate_encounter_surrogate_data_v3_rollout_local.py",
        purpose=(
            "Final rollout-local data generator. Creates targeted encounter "
            "windows around the real IC1 event near t=82.56 yr."
        ),
    ),
    SelectedFile(
        group="code",
        source_rel="train_encounter_surrogate_v2_velocitysafe_fixed2.py",
        dest_rel="01_code/train_encounter_surrogate_v2_velocitysafe_fixed2.py",
        purpose=(
            "Final velocity-safe residual-model trainer used for the successful "
            "rollout-local EncounterNN model. Included as the final training code."
        ),
    ),
    SelectedFile(
        group="code",
        source_rel="diagnose_single_event_replay_v1.py",
        dest_rel="01_code/diagnose_single_event_replay_v1.py",
        purpose=(
            "Single-event replay diagnostic. Tests whether the trained model "
            "corrects the exact t=82.56 yr encounter locally."
        ),
    ),
    SelectedFile(
        group="code",
        source_rel="diagnose_event_nearest_neighbors_v1.py",
        dest_rel="01_code/diagnose_event_nearest_neighbors_v1.py",
        purpose=(
            "Nearest-neighbour coverage diagnostic. Used to show why the earlier "
            "training data did not cover the real rollout event well enough."
        ),
    ),
    SelectedFile(
        group="code",
        source_rel="diagnose_rollout_error_timeline_v1c.py",
        dest_rel="01_code/diagnose_rollout_error_timeline_v1c.py",
        purpose=(
            "Post-event alpha/timeline diagnostic. Tests whether partial correction "
            "removes the later chaotic branch divergence."
        ),
    ),

    # Final / important result files
    SelectedFile(
        group="result",
        source_rel="encounterNN_ablation_v1_out/encounterNN_ablation_summary.txt",
        dest_rel="02_results/final_ablation/encounterNN_ablation_summary.txt",
        purpose=(
            "Final ablation result summary. This is the most important result file: "
            "it compares SIMON-noNN, SIMON-scalarNN, and SIMON-encounterNN."
        ),
    ),
    SelectedFile(
        group="result",
        source_rel="encounterNN_ablation_v1_out/encounter_events.csv",
        dest_rel="02_results/final_ablation/encounter_events.csv",
        purpose=(
            "Event log for the final ablation. Shows the gated EncounterNN event "
            "that was actually used in the 100-year rollout."
        ),
    ),
    SelectedFile(
        group="result",
        source_rel="encounter_surrogate_v3_rollout_local_velocitysafe_summary.txt",
        dest_rel="02_results/final_model_training/encounter_surrogate_v3_rollout_local_velocitysafe_summary.txt",
        purpose=(
            "Final model training/evaluation summary for the rollout-local "
            "velocity-safe EncounterNN model."
        ),
    ),
    SelectedFile(
        group="result",
        source_rel="single_event_replay_v3_rollout_local_t82p56/single_event_replay_summary.txt",
        dest_rel="02_results/single_event_replay/single_event_replay_summary.txt",
        purpose=(
            "Local replay summary for the exact t=82.56 yr event. Shows that the "
            "rollout-local model fixes the local encounter exit state."
        ),
    ),
    SelectedFile(
        group="result",
        source_rel="single_event_replay_v3_rollout_local_t82p56/single_event_replay_metrics.csv",
        dest_rel="02_results/single_event_replay/single_event_replay_metrics.csv",
        purpose=(
            "CSV metrics corresponding to the single-event replay summary."
        ),
    ),
    SelectedFile(
        group="result",
        source_rel="rollout_error_timeline_v1c/rollout_error_timeline_summary.txt",
        dest_rel="02_results/branch_timeline/rollout_error_timeline_summary.txt",
        purpose=(
            "Timeline diagnostic summary. Explains when the EncounterNN trajectory "
            "becomes worse than noNN after the corrected encounter."
        ),
    ),
    SelectedFile(
        group="result",
        source_rel="rollout_error_timeline_v1c/alpha_sweep_summary.csv",
        dest_rel="02_results/branch_timeline/alpha_sweep_summary.csv",
        purpose=(
            "Alpha sweep summary. Shows that alpha=1.0 was best among tested "
            "partial-correction strengths, but branch sensitivity remained."
        ),
    ),

    # Optional high-level report, only copied if you manually place it in this folder.
    SelectedFile(
        group="report_optional",
        source_rel="EncounterNN_Mentor_Report.pdf",
        dest_rel="00_report/EncounterNN_Mentor_Report.pdf",
        purpose=(
            "Optional comprehensive narrative report. Copied only if this PDF exists "
            "inside encounter_nn."
        ),
        required=False,
    ),
]


def is_excluded_binary(path: Path) -> bool:
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def safe_remove_folder(path: Path, base: Path) -> None:
    """Delete destination folder only if it is exactly inside the base folder."""
    path = path.resolve()
    base = base.resolve()
    if path.name != DEST_FOLDER_NAME:
        raise RuntimeError(f"Refusing to remove unexpected folder: {path}")
    if path.parent != base:
        raise RuntimeError(f"Refusing to remove folder outside base: {path}")
    if path.exists():
        shutil.rmtree(path)


def copy_one(base: Path, dest_root: Path, item: SelectedFile) -> dict:
    src = base / item.source_rel
    dst = dest_root / item.dest_rel

    row = {
        "group": item.group,
        "status": "",
        "size_bytes": "",
        "source": str(src),
        "destination": str(dst),
        "purpose": item.purpose,
    }

    if is_excluded_binary(src):
        row["status"] = "SKIPPED_BINARY_MODEL_OR_DATA"
        return row

    if not src.exists():
        row["status"] = "MISSING_REQUIRED" if item.required else "NOT_PRESENT_OPTIONAL"
        return row

    if src.is_dir():
        row["status"] = "SKIPPED_DIRECTORY_UNEXPECTED"
        return row

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    row["status"] = "COPIED"
    row["size_bytes"] = str(dst.stat().st_size)
    return row


def write_readme(dest_root: Path, manifest_rows: List[dict]) -> None:
    copied = [r for r in manifest_rows if r["status"] == "COPIED"]
    missing_required = [r for r in manifest_rows if r["status"] == "MISSING_REQUIRED"]
    optional_absent = [r for r in manifest_rows if r["status"] == "NOT_PRESENT_OPTIONAL"]

    readme = dest_root / "README_FOR_MENTOR.txt"
    with readme.open("w", encoding="utf-8") as f:
        f.write("EncounterNN Mentor Review Package\n")
        f.write("=" * 80 + "\n\n")

        f.write("Purpose\n")
        f.write("-" * 80 + "\n")
        f.write(
            "This folder contains a small curated set of files for reviewing the "
            "EncounterNN experiment. It intentionally excludes PyTorch model files "
            "(.pt) and NumPy datasets (.npz/.npy), because those are binary files and "
            "are not useful for quick mentor review.\n\n"
        )

        f.write("What this package is meant to show\n")
        f.write("-" * 80 + "\n")
        f.write(
            "1. Why the EncounterNN path was tested: a window-level residual correction "
            "was intended to improve difficult close-encounter exit states.\n"
            "2. What finally worked locally: rollout-local data around the real IC1 "
            "event allowed the model to correct the local t=82.56 yr encounter.\n"
            "3. What did not work globally: in the full 100-year ablation, EncounterNN "
            "did not beat the scalar Zone-3 NN on the main position time-average metric "
            "and was slower because it still had to run the local noNN window.\n\n"
        )

        f.write("Folder structure\n")
        f.write("-" * 80 + "\n")
        f.write("00_report/   Optional narrative report, if present.\n")
        f.write("01_code/     Main code files only.\n")
        f.write("02_results/  Main text/CSV result files only.\n")
        f.write("MANIFEST.csv Machine-readable list of copied/missing files.\n\n")

        f.write("Included files and what each file is\n")
        f.write("-" * 80 + "\n")
        for r in copied:
            rel = Path(r["destination"]).relative_to(dest_root)
            f.write(f"\n{rel}\n")
            f.write(f"  Type   : {r['group']}\n")
            f.write(f"  Purpose: {r['purpose']}\n")

        if missing_required:
            f.write("\n\nRequired files that were NOT found\n")
            f.write("-" * 80 + "\n")
            f.write(
                "These files were selected for the package but were not found in the "
                "current encounter_nn folder. Check whether they are in another folder "
                "or whether the output was saved under a different name.\n"
            )
            for r in missing_required:
                f.write(f"\n{r['source']}\n")
                f.write(f"  Expected purpose: {r['purpose']}\n")

        if optional_absent:
            f.write("\n\nOptional files not present\n")
            f.write("-" * 80 + "\n")
            for r in optional_absent:
                f.write(f"\n{r['source']}\n")
                f.write(f"  Optional purpose: {r['purpose']}\n")

        f.write("\n\nWhy the earlier MANIFEST had many missing files\n")
        f.write("-" * 80 + "\n")
        f.write(
            "The earlier package script tried to collect too many intermediate files, "
            "including superseded code names, intermediate diagnostics, model/data "
            "binaries, and report files that were not actually stored inside "
            "C:\\Aarush\\Physics\\training\\encounter_nn. This v2 script fixes that "
            "by selecting only the main review files and by not requesting known "
            "superseded files such as v1/v1b timeline scripts or old merge/inspect "
            "helpers.\n"
        )

        f.write("\n\nRecommended reading order\n")
        f.write("-" * 80 + "\n")
        f.write("1. 02_results/final_ablation/encounterNN_ablation_summary.txt\n")
        f.write("2. 02_results/final_ablation/encounter_events.csv\n")
        f.write("3. 02_results/single_event_replay/single_event_replay_summary.txt\n")
        f.write("4. 02_results/branch_timeline/rollout_error_timeline_summary.txt\n")
        f.write("5. 01_code/pair_eval_encounterNN_ablation_v1.py\n")
        f.write("6. 01_code/generate_encounter_surrogate_data_v3_rollout_local.py\n")
        f.write("7. 01_code/train_encounter_surrogate_v2_velocitysafe_fixed2.py\n")


def write_manifest(dest_root: Path, rows: List[dict]) -> None:
    manifest_path = dest_root / "MANIFEST.csv"
    fieldnames = ["group", "status", "size_bytes", "source", "destination", "purpose"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a small EncounterNN mentor-review package without .pt/.npz files."
    )
    parser.add_argument(
        "--base",
        default=".",
        help="Base folder. Default is current folder. Run from encounter_nn unless you pass this explicitly.",
    )
    parser.add_argument(
        "--dest",
        default=DEST_FOLDER_NAME,
        help="Destination folder name. Default: 'send to mentor'.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the existing destination folder first. Default is to clean it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied, but do not copy files.",
    )
    args = parser.parse_args()

    base = Path(args.base).resolve()
    dest_root = (base / args.dest).resolve()

    if not base.exists():
        raise FileNotFoundError(f"Base folder does not exist: {base}")

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN: no files will be copied")
        print(f"Base: {base}")
        print(f"Dest: {dest_root}")
        print("=" * 80)
        for item in SELECTED_FILES:
            src = base / item.source_rel
            status = "FOUND" if src.exists() else ("MISSING_REQUIRED" if item.required else "NOT_PRESENT_OPTIONAL")
            excluded = " EXCLUDED_BINARY" if is_excluded_binary(src) else ""
            print(f"{status:22s}{excluded:18s} {item.source_rel}")
        return

    if not args.no_clean:
        safe_remove_folder(dest_root, base)

    dest_root.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for item in SELECTED_FILES:
        rows.append(copy_one(base, dest_root, item))

    write_manifest(dest_root, rows)
    write_readme(dest_root, rows)

    copied = sum(1 for r in rows if r["status"] == "COPIED")
    missing = sum(1 for r in rows if r["status"] == "MISSING_REQUIRED")
    optional_absent = sum(1 for r in rows if r["status"] == "NOT_PRESENT_OPTIONAL")

    print("=" * 80)
    print("EncounterNN mentor package prepared")
    print(f"Base folder : {base}")
    print(f"Output folder: {dest_root}")
    print(f"Copied files: {copied}")
    print(f"Missing required files: {missing}")
    print(f"Optional files not present: {optional_absent}")
    print("=" * 80)
    print(f"Read: {dest_root / 'README_FOR_MENTOR.txt'}")
    print(f"Manifest: {dest_root / 'MANIFEST.csv'}")

    if missing:
        print("\nWARNING: Some required files were not found. Open MANIFEST.csv or README_FOR_MENTOR.txt.")
    else:
        print("\nNo required selected files are missing.")


if __name__ == "__main__":
    main()
