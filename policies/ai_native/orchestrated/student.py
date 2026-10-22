"""Uncached whole-chunk access to the existing OpenPI DROID wire contract."""

import numpy as np

from policies.pi0_family.client import Pi0DroidJointposClient


class Pi05ProposalClient(Pi0DroidJointposClient):
    def propose(self, raw_obs, instruction):
        extracted = self._extract_observation(raw_obs, env_id=0)
        response = self._query_server(self._pack_request(extracted, instruction))
        chunk = np.asarray(self._unpack_response(response), dtype=np.float32)
        if chunk.shape != (15, 8) or not np.isfinite(chunk).all():
            raise ValueError("Orchestrated requires a finite pi05 [15, 8] absolute joint proposal")
        return self._postprocess_chunk(chunk)

    def close(self):
        # OpenPI's client currently exposes its websocket rather than a close hook.
        socket = getattr(self.client, "_ws", None)
        if socket is not None:
            socket.close()
