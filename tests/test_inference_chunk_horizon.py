# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed execution horizons and full-chunk execution share the inference loop."""

from collections import Counter

import numpy as np
import pytest

from robolab.eval.base_client import InferenceClient


class ChunkClient(InferenceClient):
    def __init__(self, horizon, chunks):
        super().__init__()
        self.open_loop_horizon = horizon
        self.responses = {eid: iter(values) for eid, values in chunks.items()}
        self.calls = Counter()

    def _extract_observation(self, raw_obs, *, env_id=0):
        return {'env_id': env_id}

    def _pack_request(self, extracted_obs, instruction):
        return extracted_obs['env_id']

    def _query_server(self, env_id):
        self.calls[env_id] += 1
        return np.asarray(next(self.responses[env_id]), dtype=float).reshape(-1, 1)

    def _unpack_response(self, response):
        return response


def test_fixed_horizon_replans_before_long_response_is_exhausted():
    client = ChunkClient(2, {0: [[10, 11, 12, 13], [20, 21, 22]]})
    actions = [client.infer({}, 'task')['action'][0] for _ in range(3)]
    assert actions == [10, 11, 20]
    assert client.calls == {0: 2}


def test_none_consumes_successive_chunks_of_different_lengths():
    client = ChunkClient(None, {0: [[10, 11], [20, 21, 22], [30]]})
    actions = [client.infer({}, 'task')['action'][0] for _ in range(6)]
    assert actions == [10, 11, 20, 21, 22, 30]
    assert client.calls == {0: 3}


def test_batch_refresh_and_reset_are_independent_per_environment():
    client = ChunkClient(None, {0: [[10], [11, 12], [13]], 1: [[20, 21, 22]]})
    first = client.infer_batch({}, 'task', env_ids=[0, 1])
    second = client.infer_batch({}, 'task', env_ids=[0, 1])
    assert [first[i]['action'][0] for i in (0, 1)] == [10, 20]
    assert [second[i]['action'][0] for i in (0, 1)] == [11, 21]
    assert client.calls == {0: 2, 1: 1}
    client.reset(env_id=0)
    third = client.infer_batch({}, 'task', env_ids=[0, 1])
    assert [third[i]['action'][0] for i in (0, 1)] == [13, 22]
    assert client.calls == {0: 3, 1: 1}


@pytest.mark.parametrize('horizon', [0, -1, 'MAX', 1.5, True])
def test_invalid_horizon_is_rejected_before_query(horizon):
    client = ChunkClient(horizon, {0: [[1]]})
    with pytest.raises(ValueError, match='positive integer or None'):
        client.infer({}, 'task')
    assert not client.calls


@pytest.mark.parametrize('horizon', [None, 2])
def test_empty_response_is_rejected_before_caching(horizon):
    client = ChunkClient(horizon, {0: [[]]})
    with pytest.raises(ValueError, match='at least one action'):
        client.infer({}, 'task')
    assert not client._chunks
