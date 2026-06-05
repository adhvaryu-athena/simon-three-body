EncounterNN Mentor Review Package
================================================================================

Purpose
--------------------------------------------------------------------------------
This folder contains a small curated set of files for reviewing the EncounterNN experiment. It intentionally excludes PyTorch model files (.pt) and NumPy datasets (.npz/.npy), because those are binary files and are not useful for quick mentor review.

What this package is meant to show
--------------------------------------------------------------------------------
1. Why the EncounterNN path was tested: a window-level residual correction was intended to improve difficult close-encounter exit states.
2. What finally worked locally: rollout-local data around the real IC1 event allowed the model to correct the local t=82.56 yr encounter.
3. What did not work globally: in the full 100-year ablation, EncounterNN did not beat the scalar Zone-3 NN on the main position time-average metric and was slower because it still had to run the local noNN window.

Folder structure
--------------------------------------------------------------------------------
00_report/   Optional narrative report, if present.
01_code/     Main code files only.
02_results/  Main text/CSV result files only.
MANIFEST.csv Machine-readable list of copied/missing files.

Included files and what each file is
--------------------------------------------------------------------------------

01_code\pair_eval_encounterNN_ablation_v1.py
  Type   : code
  Purpose: Final complete-simulation ablation evaluator. Compares IAS15, SIMON-noNN, SIMON-scalarNN, and SIMON-encounterNN at dt=0.08.

01_code\generate_encounter_surrogate_data_v3_rollout_local.py
  Type   : code
  Purpose: Final rollout-local data generator. Creates targeted encounter windows around the real IC1 event near t=82.56 yr.

01_code\train_encounter_surrogate_v2_velocitysafe_fixed2.py
  Type   : code
  Purpose: Final velocity-safe residual-model trainer used for the successful rollout-local EncounterNN model. Included as the final training code.

01_code\diagnose_single_event_replay_v1.py
  Type   : code
  Purpose: Single-event replay diagnostic. Tests whether the trained model corrects the exact t=82.56 yr encounter locally.

01_code\diagnose_event_nearest_neighbors_v1.py
  Type   : code
  Purpose: Nearest-neighbour coverage diagnostic. Used to show why the earlier training data did not cover the real rollout event well enough.

01_code\diagnose_rollout_error_timeline_v1c.py
  Type   : code
  Purpose: Post-event alpha/timeline diagnostic. Tests whether partial correction removes the later chaotic branch divergence.

02_results\final_ablation\encounterNN_ablation_summary.txt
  Type   : result
  Purpose: Final ablation result summary. This is the most important result file: it compares SIMON-noNN, SIMON-scalarNN, and SIMON-encounterNN.

02_results\final_ablation\encounter_events.csv
  Type   : result
  Purpose: Event log for the final ablation. Shows the gated EncounterNN event that was actually used in the 100-year rollout.

02_results\final_model_training\encounter_surrogate_v3_rollout_local_velocitysafe_summary.txt
  Type   : result
  Purpose: Final model training/evaluation summary for the rollout-local velocity-safe EncounterNN model.

02_results\single_event_replay\single_event_replay_summary.txt
  Type   : result
  Purpose: Local replay summary for the exact t=82.56 yr event. Shows that the rollout-local model fixes the local encounter exit state.

02_results\single_event_replay\single_event_replay_metrics.csv
  Type   : result
  Purpose: CSV metrics corresponding to the single-event replay summary.

02_results\branch_timeline\rollout_error_timeline_summary.txt
  Type   : result
  Purpose: Timeline diagnostic summary. Explains when the EncounterNN trajectory becomes worse than noNN after the corrected encounter.

02_results\branch_timeline\alpha_sweep_summary.csv
  Type   : result
  Purpose: Alpha sweep summary. Shows that alpha=1.0 was best among tested partial-correction strengths, but branch sensitivity remained.


Optional files not present
--------------------------------------------------------------------------------

C:\Aarush\Physics\training\encounter_nn\EncounterNN_Mentor_Report.pdf
  Optional purpose: Optional comprehensive narrative report. Copied only if this PDF exists inside encounter_nn.


Why the earlier MANIFEST had many missing files
--------------------------------------------------------------------------------
The earlier package script tried to collect too many intermediate files, including superseded code names, intermediate diagnostics, model/data binaries, and report files that were not actually stored inside C:\Aarush\Physics\training\encounter_nn. This v2 script fixes that by selecting only the main review files and by not requesting known superseded files such as v1/v1b timeline scripts or old merge/inspect helpers.


Recommended reading order
--------------------------------------------------------------------------------
1. 02_results/final_ablation/encounterNN_ablation_summary.txt
2. 02_results/final_ablation/encounter_events.csv
3. 02_results/single_event_replay/single_event_replay_summary.txt
4. 02_results/branch_timeline/rollout_error_timeline_summary.txt
5. 01_code/pair_eval_encounterNN_ablation_v1.py
6. 01_code/generate_encounter_surrogate_data_v3_rollout_local.py
7. 01_code/train_encounter_surrogate_v2_velocitysafe_fixed2.py
