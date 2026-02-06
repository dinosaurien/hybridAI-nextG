from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from ain.common.types import ControlAction, Playbook

# ---- Actor ----

class Actor:
    """Converts OTM objects to structured JSON and saves them as OTM files."""


# If you already have your own ControlAction/Playbook classes, you can still use 
# Actor.make_payload() — it only requires that playbook.actions contain items with 
# a to_dict() method returning the same keys. (Good GPT prompt i think)