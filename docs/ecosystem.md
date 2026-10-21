# Ecosystem

Projects and repositories that build on RoboLab — task libraries, policy
backends, and research projects that use RoboLab for evaluation.

Tasks are plain Python dataclasses and inference clients are self-contained,
so both can live in external repositories and plug into RoboLab without
forking it. See [Creating New Tasks](task.md) and
[Evaluating a New Policy](policy.md) for how to build your own.

## Task Libraries

| Task Library | # Tasks | Objective | Notes |
|---|---:|---|---|
| [RoboVoLo](https://github.com/NVlabs/RoboVoLo) | [126](https://github.com/NVlabs/RoboVoLo#robovolo) | Reasoning, Long-Horizon | Task library from the [VoLo](https://chicychen.github.io/VoLo/) project, extending RoboLab with long-horizon and reasoning-heavy manipulation tasks: multi-step sorting and restacking, spatial reference, recycling, and math/chemistry cube puzzles. Evaluated through the [VoLo policy backend](../policies/volo/README.md) bundled with RoboLab. |
| [LADDER](https://ladder.insait.ai/) | [44](https://ladder.insait.ai/#benchmark) | Long-Horizon | RoboLab-Long suite from [LADDER](https://github.com/insait-institute/ladder), evaluating increasing task horizons in the same scene. The full benchmark contains 145 tasks across three simulation suites, with 1–10 subgoals per instruction. |

## Adding your project

Built a task library, policy client, or research project on RoboLab? Open a
pull request adding a row to the table above — see
[CONTRIBUTING.md](../CONTRIBUTING.md).
