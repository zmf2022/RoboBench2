# Astranomy: AI-Native Robot Policies

This directory contains the RoboLab policy implementations for [Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/), an empirical study of GPT-6 Astra as a native robot policy. Astranomy studies direct action generation, robot control through a persistent agent conversation, and model supervision of a robot foundation policy.

For environment installation and backend selection, start with the
[Astranomy setup guide](../../README.md#astranomy).

## Policies

| Policy | Description | Status | Documentation |
|---|---|---|---|
| Direct-API (Direct Policy) | Astra predicts robot actions directly through API calls. | Available | [Direct-API README](direct_api/README.md) |
| Agentic | Astra controls the robot through Codex, with conversation history and optional tools. | Available | [Agentic README](agentic/README.md) |
| Orchestrated | Astra reviews fresh π₀.₅ proposals through Codex or direct API, using the original student/edit/eef/stop contract. | Available; live validation pending | [Orchestrated README](orchestrated/README.md) |

Each policy's README covers its setup, supported features, configuration, and usage.
