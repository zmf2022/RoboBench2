# Scripts

Repository helpers are grouped below by purpose. Run Python commands from the repository
root with the project environment activated.

## Environment setup

`./scripts/setup_env.sh` installs Isaac Sim 6.0.1 with Python 3.12 in the project-local `.venv`.
See [installation](../README.md#installation) for prerequisites, assets and checks.
Use `--help` to inspect its options without installing anything.

## Model servers

| Script | Purpose |
| --- | --- |
| `install_openpi.sh`, `serve_pi05.sh` | Install OpenPI and serve the π0.5 joint-position checkpoint |
| `start_cosmos3.sh`, `start_cosmos3_policy_server.py` | Launch the Cosmos3 policy server |
| `start_qwen3.sh` | Launch the Qwen3 server |

Backend-specific dependencies and usage belong in the [policy documentation](../policies/README.md).
Astranomy launchers live under [`policies/ai_native/`](../policies/ai_native/README.md).

## Offline evaluation and analysis

The tools in this table do not need Isaac Sim. `h5py` tools read recorded HDF5 trajectories;
`usd-core` tools inspect USD assets.

| script | what it does | needs |
|---|---|---|
| `verify_patches.py` | Per recorded run, PASS / FAIL / N/A for each evaluation change; `--baseline` for before/after, `--summary` across runs, exit 1 on FAIL. The evidence behind [changes.md](../docs/verified/changes.md). | h5py (P77) |
| `reflag.py` | Replay the post-processable event rules over an episode already on disk. | — |
| `flag_regression.py` | Score the flags of recorded runs against the human verdicts in `analysis/flag_labels.jsonl`. | — |
| `friction_sweep_report.py` | The per-condition table of a `--friction` sweep ([physics.md](../docs/physics.md)). | h5py (optional) |
| `contact_force_profile.py` | Pad-force statistics over confirmed carries vs failed attempts (the tow question). | h5py |
| `audit_task_definitions.py` | Every task definition against its scene and spawn state; exit 1 on a finding. | usd-core |
| `find_task_definition_conflicts.py` | AST diff of each task's success placement against its subtask ladder. | — |
| `check_scene_intersections.py` | Objects authored overlapping each other; differential by design. | usd-core |
| `check_rest_heights.py` | Objects that drop, rise or roll at reset, from recordings. | h5py |
| `find_sinking_objects.py` | Objects that sink into their support at reset. | h5py |
| `find_open_hand_carries.py` | Candidates for the "stuck to a finger" artifact. | h5py |
| `flag_tows.py` | The towing artifact flag over recorded runs ([towing.md](../docs/verified/towing.md)): writes `tows_<run>_env<env>.json` per episode, prints tier-A / A-or-B prevalence per run, `--csv` for a review list, `--summary` to re-read existing files. | h5py, usd-core |
| `grip_before_carry.py` | Checks the `OBJECT_GRIPPED → OBJECT_CARRIED` ordering in recorded logs. | — |
| `annotate_phases.py` | Dense phase annotation of recorded runs ([dense_annotations.md](../docs/verified/dense_annotations.md)): one label per step plus pick / place / drop segments, written as `phases_<run>_env<env>.json`; `--summary`, `--check-events` (H2), `--check-replay` (H5), `--show`. | h5py, torch (tracker replay) |
| `select_gold_episodes.py` | Stratified, seeded choice of the episodes for the dense-annotation gold set, with dashboard deep links; writes `analysis/gold_set.jsonl`. | — |
| `score_gold.py` | Scores the gold labels: annotator against annotator (H6) and the machine's segments against the consensus (H8); segment F1 at IoU 0.5, boundary recall at ±0.25 / 0.5 / 1 / 2 s, recall by duration. | h5py, torch (when it annotates on the fly) |
| `read_subtask_status_from_hdf5.py` | Dump the per-step subtask status from a recording. | h5py |
| `convert_to_lerobot.py` | Export recordings to the LeRobot format (upstream). | — |

Usage lines: `python scripts/<name>.py --help`. See [verification.md](../docs/verified/verification.md)
for how these fit together.

## Additional diagnostics

- `list_tasks.py`, `preflight_tasks.py`: task inventory and static preflight checks.
- `check_dense_channels.py`, `phase_mode_check.py`, `phase_sensitivity.py`,
  `phase_stability.py`, `reflag_phases.py`: dense phase annotation diagnostics.
- `h7_log_disagreement.py`, `policy_failure_taxonomy.py`: disagreement and failure analysis.
- [`_shims/`](_shims/README.md): offline import shims used by tooling.

Read each script's `--help` for its inputs and dependencies. Historical evidence is indexed in
[verification](../docs/verified/verification.md).
