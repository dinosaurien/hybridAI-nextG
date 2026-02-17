from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

@dataclass
class Msg:
    topic: str
    type: str
    ts: str
    id: str
    corr_id: str
    schema: str
    payload: dict

def make_msg(topic, type_, schema, payload, corr_id=None):
    return Msg(
        topic=topic,
        type=type_,
        ts=datetime.now(timezone.utc).isoformat(),
        id=str(uuid.uuid4()),
        corr_id=corr_id or str(uuid.uuid4()),
        schema=schema,
        payload=payload,
    )