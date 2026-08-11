"""Wire-shape conversion for the evaluation endpoints.

The evaluation service returns the frozen contract
:class:`EvaluationGetResponseData` (task snapshot + params + metric);
the endpoints wrap that snapshot in the standard ``{success, data}``
envelope, mirroring the upstream handler's response shape. No service
DTO sits between the service and the wire: the service already emits
the frozen contract, so this module only carries the envelope.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.contracts.evaluation import EvaluationGetResponseData


class EvaluationEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - evaluation task responses.

    Used by both ``POST /evaluation`` (the freshly created task
    snapshot) and ``GET /evaluation`` (the latest snapshot for a task
    id). The envelope shape matches the rest of the API.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EvaluationGetResponseData


def evaluation_envelope(data: EvaluationGetResponseData) -> EvaluationEnvelope:
    """Wrap an evaluation snapshot in the success envelope."""
    return EvaluationEnvelope(success=True, data=data)


__all__ = ["EvaluationEnvelope", "evaluation_envelope"]
