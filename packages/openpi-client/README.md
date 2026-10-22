# Bundled OpenPI client

This package is copied from `Physical-Intelligence/openpi` at commit
`00a968d05b936af959607218a3f1d0bad58a9a44`. Only five runtime files are bundled:

- `__init__.py`: package entry point.
- `base_policy.py`: used by the client and the local policy-server shim.
- `websocket_client_policy.py`: existing WebSocket request/response transport.
- `msgpack_numpy.py`: NumPy-aware wire codec, also used by the server shim.
- `image_tools.py`: the π₀ family client's image preprocessing.

The unused action chunk broker and runtime helpers are omitted, together with
their `dm-tree` dependency. The OpenPI/JAX server and checkpoints remain separate.

The metadata removes the upstream `numpy<2` cap and explicitly declares
`typing-extensions`, which the source imports. The codec explicitly rejects NumPy
2 variable-width `StringDType` before encoding: its managed string storage cannot
be represented by the existing dtype + raw bytes wire format. Numeric arrays,
NumPy scalars, fixed-width strings and image data keep the upstream wire format.
No dtype conversion or NumPy downgrade is applied to robot observations/actions.
Other runtime Python source is unchanged. The upstream Apache-2.0 license is
included in the repository's [LICENSE](../../LICENSE).

From the RoboBench2 repository root:

```bash
uv sync --extra isaac60 --extra test
```

The root `pyproject.toml` selects this local editable package through
`tool.uv.sources`; a separate OpenPI checkout is not needed to install the client.
For an existing environment, install only this client with:

```bash
uv pip install --python .venv/bin/python -e packages/openpi-client
```

When updating, copy source files from the intended upstream revision and update
the revision above. Retain and document any local
changes instead of silently mixing upstream versions.
