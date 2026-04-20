import asyncio
import copy
import json
import time
import logging
import uuid as _uuid
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional
from core.bus.mem import MemBus
from core.bus.messages import make_msg
from core.utils.knowledge_base import KnowledgeBase
from core.common.types import ScheduledIntent, ScheduleState, ActiveProcedure, ProcedureState, ProcedureStep
from collections import deque

logger = logging.getLogger(__name__)

class IntentState(Enum):
    MONITORING = auto()
    THINKING = auto()
    ASSURANCE = auto()  # 30s grace period for constraint executor to apply actions
    WITHDRAWAL = auto() # Evaluate whether the OTM resolved the deviation
    ESCALATED = auto()  # System frozen after repeated failures — awaiting human intervention

class OrchestratorAgent:
    THINKING_TIMEOUT = 75.0   # seconds before assuming LLM failed
    WITHDRAWAL_WINDOW = 10.0  # seconds to evaluate post-assurance telemetry
    MAX_ADAPTATION_CYCLES = 5  # cap adaptation loops before declaring failure
    ESCALATED_TIMEOUT = 300.0  # seconds before auto-unlocking ESCALATED state
    HEALTH_CHECK_WINDOW = 10.0 # seconds of telemetry data to consider for health check

    def __init__(self, bus: MemBus, kb: KnowledgeBase, episode_store=None):
        self.bus = bus
        self.kb = kb
        self.episode_store = episode_store
        self.state = IntentState.MONITORING
        self.thinking_start_time = 0.0
        self.assurance_end_time = 0.0
        self.withdrawal_end_time = 0.0
        self.active_otm = None
        self.active_anomaly_metric = None
        self.active_cell_id: Optional[str] = None  # Cell scope of the current anomaly
        self.active_ue_id: Optional[str] = None    # UE scope (if UE_* metric)
        self.triggering_anomaly: Optional[dict] = None  # Full anomaly payload for episode recording
        self.pending_manual_intents = []
        self.e2_acks_received = 0
        self.last_metric_value = None
        self.schedule_queue: List[ScheduledIntent] = []
        self.active_procedure: Optional[ActiveProcedure] = None
        self.pending_cancel_revert = None  # OTM to revert to after graceful cancel completes
        self.last_kpi_snapshot: dict = {}  # Full latest telemetry for health checks
        self.kpi_history = deque() # Stores timestamps and snapshots of recent KPIs for health check evaluation
        self.scenario_failure_count: int = 0  # Consecutive scenario failures (for escalation)
        self.adaptation_cycle_count: int = 0  # Consecutive adapt_otm cycles for current anomaly
        self.escalated_at: float = 0.0  # Timestamp when ESCALATED was entered
        self.escalated_anomaly_queue: list = []  # Anomalies received while ESCALATED

        # Evaluation event log, append-only JSONL for later analysis of orchestration logic
        self._event_log_path = Path("models/orchestrator_events.jsonl")
        self._event_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_event(self, event_type: str, **data):
        """Append a structured event to the orchestrator event log."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "state": self.state.name,
            **data,
        }
        try:
            with open(self._event_log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Never crash for logging

    async def run(self):
        q_dev = await self.bus.sub("deviation.detected")
        q_otm = await self.bus.sub("ai.response")
        q_ui  = await self.bus.sub("ui.input")
        q_cmd = await self.bus.sub("command.notify")
        q_kpi = await self.bus.sub("kpi.raw")
        q_cancel = await self.bus.sub("schedule.cancel")

        logger.info("[ORCHESTRATOR] Orchestrator Online. Monitoring for Anomalies & Manual Intents.")

        while True:
            current_time = time.time()

            # Drain E2 acks (track that commands were dispatched)
            while not q_cmd.empty():
                q_cmd.get_nowait()
                self.e2_acks_received += 1

            # Track latest value of the anomaly metric from telemetry
            while not q_kpi.empty():
                kpi_msg = q_kpi.get_nowait()
                raw = kpi_msg.payload.get("kpi", kpi_msg.payload)
                
                self.last_kpi_snapshot = raw  # Keep for backwards compatibility
                
                # Append to rolling history and prune
                now = time.time()
                self.kpi_history.append((now, raw))
                
                while self.kpi_history and now - self.kpi_history[0][0] > self.HEALTH_CHECK_WINDOW:
                    self.kpi_history.popleft()
                
                if self.active_anomaly_metric:
                    metric = self.active_anomaly_metric
                    val = None
                    if metric.startswith("UE_"):
                        target_ue = None
                        if self.triggering_anomaly:
                            target_ue = (self.triggering_anomaly.get("scope") or {}).get("ue_id")
                        if target_ue is not None:
                            target_ue = str(target_ue)
                        for ue in raw.get("UEMetrics", []) or []:
                            if target_ue is None or str(ue.get("ue_id")) == target_ue:
                                if metric in ue:
                                    val = ue[metric]
                                    break
                    else:
                        val = raw.get("CellMetrics", {}).get(metric)
                    if val is not None:
                        self.last_metric_value = float(val)

            # Timeout: if LLM hasn't responded, unlock the loop
            if self.state == IntentState.THINKING and current_time >= self.thinking_start_time + self.THINKING_TIMEOUT:
                logger.warning("[ORCHESTRATOR] THINKING timeout reached — Cognitive Core did not respond. Returning to MONITORING.")
                self._clear_anomaly_state()
                self.state = IntentState.MONITORING

            # ESCALATED auto-unlock: don't stay frozen forever if no human intervenes
            if self.state == IntentState.ESCALATED and current_time >= self.escalated_at + self.ESCALATED_TIMEOUT:
                logger.warning(f"[ORCHESTRATOR] ESCALATED timeout ({self.ESCALATED_TIMEOUT}s) reached. "
                               "Auto-unlocking to MONITORING. Safe-mode OTM remains active.")
                self._log_event("escalated_timeout")
                self.state = IntentState.MONITORING

            # ASSURANCE -> WITHDRAWAL transition
            if self.state == IntentState.ASSURANCE and current_time >= self.assurance_end_time:
                if self.e2_acks_received == 0 and not (
                    self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE
                ):
                    logger.warning("[ORCHESTRATOR] Assurance expired but NO E2 commands were dispatched. Returning to MONITORING.")
                    self.state = IntentState.MONITORING
                else:
                    logger.info(f"[ORCHESTRATOR] Assurance expired ({self.e2_acks_received} E2 commands dispatched). Entering WITHDRAWAL to evaluate outcome.")
                    self.state = IntentState.WITHDRAWAL
                    self.withdrawal_end_time = current_time + self.WITHDRAWAL_WINDOW

            # WITHDRAWAL evaluation
            if self.state == IntentState.WITHDRAWAL and current_time >= self.withdrawal_end_time:
                resolved = await self._evaluate_withdrawal()
                if resolved:
                    self.adaptation_cycle_count = 0  # Reset on success
                    # If a procedure is active, advance to the next step instead of MONITORING
                    if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE:
                        self.active_procedure.step_failures = 0
                        advanced = await self._advance_procedure()
                        if not advanced:
                            # No more steps — procedure complete. Record final episode.
                            await self._record_and_reflect(resolved=True)
                            await self._complete_procedure()
                            await self._release_anomaly_otm_if_needed()
                            self._clear_anomaly_state()
                            self.state = IntentState.MONITORING
                        # else: intermediate step — _advance_procedure set state to ASSURANCE, no episode yet
                    else:
                        # No procedure — simple anomaly→OTM→resolved cycle. Record episode.
                        await self._record_and_reflect(resolved=True)
                        logger.info("[ORCHESTRATOR] WITHDRAWAL: Deviation resolved. Returning to MONITORING.")
                        await self._release_anomaly_otm_if_needed()
                        self._clear_anomaly_state()
                        self.state = IntentState.MONITORING
                else:
                    # Check procedure step failure limit
                    if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE:
                        self.active_procedure.step_failures += 1
                        if self.active_procedure.step_failures >= 3:
                            # Procedure exhausted — record failed episode before rollback
                            await self._record_and_reflect(resolved=False)
                            logger.warning(f"[ORCHESTRATOR] Procedure step failed 3 times. Rolling back procedure.")
                            await self._rollback_procedure()
                            continue

                    # Adaptation loop — don't record episode yet, wait for eventual resolution
                    self.adaptation_cycle_count += 1

                    if self.adaptation_cycle_count >= self.MAX_ADAPTATION_CYCLES:
                        # Adaptation exhausted — record failed episode, return to MONITORING
                        logger.warning(f"[ORCHESTRATOR] Adaptation limit reached ({self.MAX_ADAPTATION_CYCLES} cycles). "
                                       "Recording failure and returning to MONITORING.")
                        self._log_event("adaptation_exhausted",
                                        metric=self.active_anomaly_metric,
                                        cycles=self.adaptation_cycle_count)
                        await self._record_and_reflect(resolved=False)
                        self.adaptation_cycle_count = 0
                        self._clear_anomaly_state()
                        self.state = IntentState.MONITORING
                    else:
                        logger.warning(f"[ORCHESTRATOR] WITHDRAWAL: Deviation persists (cycle {self.adaptation_cycle_count}/{self.MAX_ADAPTATION_CYCLES}). "
                                       "Re-engaging Cognitive Core for adaptation.")
                        self.state = IntentState.THINKING
                        self.thinking_start_time = current_time
                        self.e2_acks_received = 0
                        metric = self.active_anomaly_metric
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "adapt_otm", "metric": metric,
                            "value": self.last_metric_value,
                            "previous_otm": self.active_otm,
                            "scope": (self.triggering_anomaly or {}).get("scope") or {},
                        }))

            # Graceful cancel revert: apply when system settles back to MONITORING
            if self.state == IntentState.MONITORING and self.pending_cancel_revert is not None:
                await self._apply_cancel_revert()

            while not q_ui.empty():
                msg = q_ui.get_nowait()
                text = msg.payload.get("text", "").upper()

                if "BREAK_SIM" in text:
                    action = "trigger-traffic-spike" if "LATENCY" in text else "trigger-blockage"
                    logger.info(f"[ORCHESTRATOR] Routing manual chaos command: {text} -> {action}")
                    await self.bus.pub("sim.control", make_msg("orch", "CONTROL", "v1", {"action": action}))
                else:
                    # If a procedure is actively executing, queue the manual intent
                    # to avoid orphaning the procedure mid-step
                    if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE and \
                       self.state in (IntentState.ASSURANCE, IntentState.WITHDRAWAL):
                        logger.info(f"[ORCHESTRATOR] Procedure '{self.active_procedure.scenario_name}' active. "
                                    f"Queuing manual intent: '{text}'")
                        self.pending_manual_intents.append(text)
                    elif self.state in (IntentState.MONITORING, IntentState.ASSURANCE, IntentState.WITHDRAWAL, IntentState.ESCALATED):
                        if self.state == IntentState.ESCALATED:
                            logger.info("[ORCHESTRATOR] Human override received — unlocking ESCALATED state.")
                            self.scenario_failure_count = 0
                        self.state = IntentState.THINKING
                        self.thinking_start_time = current_time
                        self.e2_acks_received = 0
                        await self._route_manual_intent(text)
                    else:
                        logger.info(f"[ORCHESTRATOR] Busy ({self.state.name}). Queuing manual intent: '{text}'")
                        self.pending_manual_intents.append(text)
                
            # Process queued manual intents the moment we become idle.
            # Placed BEFORE the anomaly queue (q_dev) so human requests take priority.
            if self.state == IntentState.MONITORING and self.pending_manual_intents:
                queued_text = self.pending_manual_intents.pop(0)
                self.state = IntentState.THINKING
                self.thinking_start_time = current_time
                self.e2_acks_received = 0
                await self._route_manual_intent(queued_text)

            # Handle schedule cancellation requests from the UI
            while not q_cancel.empty():
                cancel_msg = q_cancel.get_nowait()
                cancel_id = cancel_msg.payload.get("schedule_id")
                if cancel_id:
                    await self._cancel_scheduled_intent(cancel_id)

            while not q_dev.empty():
                msg = q_dev.get_nowait()
                dev = msg.payload
                metric = dev['metric']

                # Suppress anomaly interrupts while a procedure is actively executing —
                # the procedure IS the structured response to this anomaly
                if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE and \
                   self.state in (IntentState.ASSURANCE, IntentState.WITHDRAWAL):
                    logger.debug(f"[ORCHESTRATOR] Suppressing anomaly on {metric} — procedure "
                                 f"'{self.active_procedure.scenario_name}' is handling it.")
                    continue

                if self.state == IntentState.MONITORING:
                    logger.info(f"[ORCHESTRATOR] Anomaly Detected: {metric} (state: {self.state.name}).")
                    self.triggering_anomaly = dev  # Snapshot for episode recording
                    scope = dev.get("scope") or {}
                    self.active_cell_id = scope.get("cell_id")
                    self.active_ue_id = scope.get("ue_id")
                    self._log_event("anomaly_detected",
                                    metric=metric, value=dev.get('value'),
                                    severity=dev.get('severity', 'unknown'))
                    self.state = IntentState.THINKING
                    self.thinking_start_time = current_time
                    self.e2_acks_received = 0

                    await self.bus.pub("deviation.broadcast", make_msg("orch", "DEV", "v1", dev, corr_id=msg.corr_id))

                    anomaly_scope = dev.get("scope") or {}
                    if self.active_otm and self.active_anomaly_metric == metric:
                        # Same metric failing again: tighten existing constraints
                        logger.warning(f"[ORCHESTRATOR] Persistent failure on {metric}. Requesting ADAPTATION. (Trace: {msg.corr_id})")
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "adapt_otm", "metric": metric, "value": dev['value'],
                            "previous_otm": self.active_otm,
                            "scope": anomaly_scope,
                        }, corr_id=msg.corr_id))
                    elif self.active_otm:
                        # Different metric or manual OTM active: merge anomaly into active OTM
                        logger.info(f"[ORCHESTRATOR] Merging anomaly on {metric} into active OTM. (Trace: {msg.corr_id})")
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "merge_otm",
                            "active_otm": self.active_otm,
                            "anomaly": {
                                "metric": metric,
                                "value": dev['value'],
                                "target": dev.get('target', 40.0),
                                "direction": dev.get('direction', 'lower_better'),
                                "severity": dev.get('severity', 'medium'),
                                "scope": anomaly_scope,
                            },
                            "scope": anomaly_scope,
                        }, corr_id=msg.corr_id))
                    else:
                        # No active OTM: generate fresh
                        logger.info(f"[ORCHESTRATOR] Requesting NEW OTM from Cognitive Core. (Trace: {msg.corr_id})")
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "generate_otm", "metric": metric, "value": dev['value'],
                            "scope": anomaly_scope,
                        }, corr_id=msg.corr_id))
                    self.active_anomaly_metric = metric
                    break
                elif self.state == IntentState.ESCALATED:
                    # Queue anomalies during ESCALATED so they can be processed after unlock
                    if len(self.escalated_anomaly_queue) < 5:  # Cap to prevent OOM
                        self.escalated_anomaly_queue.append(dev)
                    logger.info(f"[ORCHESTRATOR] ESCALATED: queued anomaly on {metric} "
                                f"({len(self.escalated_anomaly_queue)} queued)")
                else:
                    logger.debug(f"[ORCHESTRATOR] Busy ({self.state.name}). Ignoring anomaly on {metric}")

            # When returning to MONITORING from ESCALATED, process the most recent queued anomaly
            if self.state == IntentState.MONITORING and self.escalated_anomaly_queue:
                latest = self.escalated_anomaly_queue[-1]
                self.escalated_anomaly_queue.clear()
                metric = latest['metric']
                logger.info(f"[ORCHESTRATOR] Processing queued anomaly from ESCALATED: {metric}")
                self.triggering_anomaly = latest
                scope = latest.get("scope") or {}
                self.active_cell_id = scope.get("cell_id")
                self.active_ue_id = scope.get("ue_id")
                self._log_event("anomaly_detected", metric=metric,
                                value=latest.get('value'), severity=latest.get('severity', 'unknown'))
                self.state = IntentState.THINKING
                self.thinking_start_time = time.time()
                self.e2_acks_received = 0
                self.active_anomaly_metric = metric

                # If a safe-mode OTM is still active from the ESCALATED path,
                # merge the new anomaly into it rather than regenerating from scratch.
                if self.active_otm:
                    await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                        "type": "merge_otm",
                        "active_otm": self.active_otm,
                        "anomaly": {
                            "metric": metric,
                            "value": latest['value'],
                            "target": latest.get('target', 40.0),
                            "direction": latest.get('direction', 'lower_better'),
                            "severity": latest.get('severity', 'medium'),
                            "scope": scope,
                        },
                        "scope": scope,
                    }))
                else:
                    await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                        "type": "generate_otm", "metric": metric,
                        "value": latest['value'],
                        "scope": scope,
                    }))

            while not q_otm.empty():
                msg = q_otm.get_nowait()
                if msg.type == "LLM_FAILURE":
                    logger.warning(f"[ORCHESTRATOR] Cognitive Core failed to generate OTM "
                                   f"(type={msg.payload.get('request_type')}). Returning to MONITORING.")
                    self._log_event("llm_generation_failed",
                                    error=msg.payload.get("error"),
                                    request_type=msg.payload.get("request_type"))
                    self._clear_anomaly_state()
                    self.state = IntentState.MONITORING

                else:
                    otm = msg.payload
                    temporal = otm.get("metadata", {}).get("temporal_resolved", {})

                    if temporal.get("has_schedule"):
                        activate_at = datetime.fromisoformat(temporal["activate_at_utc"])
                        now_utc = datetime.now(timezone.utc)

                        if activate_at > now_utc + timedelta(seconds=5):
                            # Future intent: enqueue and return to monitoring
                            await self._enqueue_scheduled_intent(otm, msg.corr_id)
                            self.state = IntentState.MONITORING
                        else:
                            # Activate immediately but track deactivation
                            if temporal.get("deactivate_at_utc"):
                                si = await self._enqueue_scheduled_intent(otm, msg.corr_id)
                                si.state = ScheduleState.ACTIVE
                                si.pre_intent_otm = self.active_otm
                            self._apply_otm(otm)
                    else:
                        self._apply_otm(otm)

            await self._tick_scheduler()
            await asyncio.sleep(0.2)

    #  Helpers

    def _clear_anomaly_state(self):
        """Reset anomaly tracking fields to prevent stale data from contaminating
        future episodes or WITHDRAWAL evaluations."""
        self.active_anomaly_metric = None
        self.active_cell_id = None
        self.active_ue_id = None
        self.triggering_anomaly = None
        self.last_metric_value = None
        self.adaptation_cycle_count = 0

    async def _route_manual_intent(self, text: str):
        """Route a manual intent — merge with active OTM if one exists, else fresh."""
        self._clear_anomaly_state()
        schedule_ctx = self._get_schedule_context()
        if self.active_otm:
            logger.info(f"[ORCHESTRATOR] Merging manual intent '{text}' with active OTM (manual takes priority).")
            await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                "type": "merge_manual", "text": text, "active_otm": self.active_otm,
                "schedule_context": schedule_ctx,
            }))
        else:
            logger.info(f"[ORCHESTRATOR] Manual User Intent: '{text}' -> Sending to AI")
            await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                "type": "manual", "text": text,
                "schedule_context": schedule_ctx,
            }))

    def _get_schedule_context(self) -> list:
        """Build a summary of PENDING/ACTIVE scheduled intents for LLM context."""
        return [
            {
                "raw_text": si.raw_text,
                "state": si.state.name,
                "activate_at": si.activate_at.isoformat(),
                "deactivate_at": si.deactivate_at.isoformat() if si.deactivate_at else None,
            }
            for si in self.schedule_queue
            if si.state in (ScheduleState.PENDING, ScheduleState.ACTIVE)
        ]

    def _apply_otm(self, otm: dict):
        """Common path: activate an OTM immediately (ASSURANCE window)."""
        # Stamp the anomaly scope into OTM metadata so downstream actuators
        # (NetworkOptimizer → ConstraintExecutor → xApp) know which cell/UE
        # this intent targets, instead of falling back to CELL_001.
        meta = otm.setdefault("metadata", {})
        if self.active_cell_id or self.active_ue_id:
            if self.active_cell_id and not meta.get("cell_id"):
                meta["cell_id"] = self.active_cell_id
            if self.active_ue_id and not meta.get("ue_id"):
                meta["ue_id"] = self.active_ue_id

        # Tag OTM source so the release path (WITHDRAWAL-resolved) can drop
        # anomaly-driven OTMs back to baseline while keeping manual/scheduled
        # intents in force.
        if not meta.get("source"):
            meta["source"] = "anomaly" if self.triggering_anomaly else "operator"

        self.active_otm = otm
        self.state = IntentState.ASSURANCE
        self.assurance_end_time = time.time() + 30.0
        self.e2_acks_received = 0

        # Initialize procedure queue if this OTM is the first step of a procedure
        procedure_id = otm.get("metadata", {}).get("procedure_id")
        if procedure_id:
            # If a different procedure is already active, cancel it first
            if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE \
               and self.active_procedure.scenario_id != procedure_id:
                prev_name = self.active_procedure.scenario_name
                self.active_procedure.state = ProcedureState.COMPLETED
                now_str = datetime.now(timezone.utc).isoformat()
                cancel_msg = (f"[{now_str}] Procedure '{prev_name}' superseded by new procedure '{procedure_id}'.")
                logger.info(f"[ORCHESTRATOR] {cancel_msg}")
                self._append_to_active_log(cancel_msg)
                self.active_procedure = None

            if self.active_procedure is None:
                steps = self.kb.get_procedure_steps(procedure_id)
                if steps:
                    self.active_procedure = ActiveProcedure(
                        scenario_id=procedure_id,
                        scenario_name=self.kb.get_scenario_name(procedure_id),
                        steps=steps,
                        current_step_idx=0,
                        pre_procedure_otm=copy.deepcopy(self.active_otm),
                    )
                    
                    first_step = steps[0]
                    self._merge_step_into_otm(first_step)
                    # forward to optimizer
                    asyncio.create_task(self._publish_procedure_update("step_activated", first_step))
                    
                    logger.info(f"[ORCHESTRATOR] Procedure '{self.active_procedure.scenario_name}' activated. Entering ASSURANCE for step 0.")

        logger.info("[ORCHESTRATOR] OTM Received. Entering 30s ASSURANCE window.")
        self._log_event("otm_applied",
                        otm=otm,
                        anomaly_metric=self.active_anomaly_metric,
                        metric_value=self.last_metric_value,
                        procedure_id=otm.get("metadata", {}).get("procedure_id"))

        asyncio.create_task(self.bus.pub("intent.execute", make_msg("orch", "NEW_TARGET", "v1", self.active_otm)))

    async def _release_anomaly_otm_if_needed(self):
        """Drop an anomaly-driven active_otm after resolution and push a
        baseline-revert OTM so ns-3 actuators return to default parameters.
        Manual/scheduled/escalated OTMs are preserved.
        """
        otm = self.active_otm
        if not otm:
            return
        source = otm.get("metadata", {}).get("source")
        if source != "anomaly":
            return

        now_str = datetime.now(timezone.utc).isoformat()
        logger.info("[ORCHESTRATOR] Anomaly resolved — releasing active_otm and reverting actuators to baseline.")

        baseline_otm = {
            "version": "1.0",
            "objective": {"service": "mbb", "kpi": "baseline", "aggregation": "mean", "unit": "", "maximize": False},
            "constraints": [
                {"service": "mbb", "kpi": "dl_mcs_max", "operator": "le", "threshold": 28, "unit": "", "id": "BASELINE_MCS", "origin": "baseline_revert"},
                {"service": "mbb", "kpi": "tx_power_dbm", "operator": "ge", "threshold": 46.0, "unit": "dBm", "id": "BASELINE_PWR", "origin": "baseline_revert"},
            ],
            "metadata": {
                "timescale": "10s_window",
                "source": "baseline_revert",
                "procedure_id": None,
                "cell_id": self.active_cell_id,
                "ue_id": self.active_ue_id,
                "adaptation_log": [f"[{now_str}] Anomaly resolved — reverting to baseline actuators."],
            },
        }

        self.active_otm = None
        await self.bus.pub("intent.execute", make_msg("orch", "BASELINE_REVERT", "v1", baseline_otm))

    async def _evaluate_withdrawal(self) -> bool:
        """Check whether the anomaly metric has recovered.

        Uses the KnowledgeBase health thresholds as the success criteria.
        Returns True if the deviation is resolved, False if it persists.

        Episode recording is handled separately by _record_and_reflect() at
        cycle boundaries (not every intermediate procedure step).
        """
        if self.active_anomaly_metric is None:
            # Nothing to evaluate (state-machine reached withdrawal without an
            # active anomaly — e.g. a scheduled-intent activation path).
            resolved = True
        elif self.last_metric_value is None:
            # Active anomaly but no telemetry in the withdrawal window — treat
            # as unresolved rather than silently concluding success.
            resolved = False
        else:
            thresholds = self.kb.get_health_thresholds(self.active_cell_id or "default")
            metric = self.active_anomaly_metric

            m = metric.lower()
            if "delay" in m or "latency" in m:
                limit = thresholds.get("latency_max_ms", 50.0)
                resolved = self.last_metric_value <= limit
            elif "thp" in m or "throughput" in m:
                # UEMetrics reports throughput in kbps (xApp emits raw ns-3
                # kbps; the adapter's *1e6 branch is not taken for most
                # telemetry paths — see kpms.csv: UE values are ~10k-20k,
                # inconsistent with bps).  Align the threshold to kbps here.
                limit = thresholds.get("throughput_min_mbps", 5.0) * 1000.0
                resolved = self.last_metric_value >= limit
            elif "bler" in m:
                limit = thresholds.get("bler_max", 0.1)
                resolved = self.last_metric_value <= limit
            else:
                resolved = False

        self._log_event("withdrawal_eval",
                        resolved=resolved,
                        metric=self.active_anomaly_metric,
                        value=self.last_metric_value,
                        e2_acks=self.e2_acks_received)

        return resolved

    async def _record_and_reflect(self, resolved: bool):
        """Record an episode at the end of an anomaly→OTM→outcome cycle and
        trigger Reflexion self-reflection.

        Only called at cycle boundaries: final resolution, procedure completion,
        or procedure rollback — NOT at every intermediate WITHDRAWAL step.
        """
        if not self.episode_store or not self.triggering_anomaly:
            return

        try:
            outcome = {
                "resolved": resolved,
                "metric_before": self.triggering_anomaly.get("value"),
                "metric_after": self.last_metric_value,
                "e2_acks": self.e2_acks_received,
            }

            # Build the OTM snapshot for the episode.
            # Use applied_value (what was actually sent to ns-3) when available,
            # falling back to threshold. This ensures Reflexion learns from
            # real outcomes, not the LLM's intended-but-biased thresholds.
            constraints_with_applied = []
            if self.active_otm:
                for c in self.active_otm.get("constraints", []):
                    c_copy = dict(c)
                    if "applied_value" in c_copy:
                        c_copy["threshold_original"] = c_copy["threshold"]
                        c_copy["threshold"] = c_copy["applied_value"]
                    constraints_with_applied.append(c_copy)

            # Pass a modified OTM copy where actuatable constraint thresholds
            # reflect what was actually applied (not the LLM's raw boundaries).
            otm_for_recording = self.active_otm
            if constraints_with_applied and self.active_otm:
                otm_for_recording = dict(self.active_otm)
                otm_for_recording["constraints"] = constraints_with_applied

            episode_id = self.episode_store.record(
                anomaly=self.triggering_anomaly,
                otm=otm_for_recording,
                outcome=outcome,
            )

            # Canonical Reflexion (Shinn et al. 2023 Algorithm 1): Msr is
            # invoked ONLY when the evaluator Me reports failure. Successful
            # trials are still recorded on disk above for audit, but we do
            # not spend LLM tokens reflecting on outcomes that need no change.
            if resolved:
                logger.info(f"[ORCHESTRATOR] Episode {episode_id[:8]} recorded "
                            f"(resolved=True, no reflection requested).")
                return

            reflect_constraints = constraints_with_applied if constraints_with_applied else []

            # Published on a separate topic so reflections don't block
            # urgent OTM generation/adaptation on the ai.request queue.
            await self.bus.pub("ai.reflect", make_msg(
                "orchestrator", "REFLECT", "v1", {
                    "type": "reflect",
                    "episode_id": episode_id,
                    "episode": {
                        "anomaly": self.triggering_anomaly,
                        "otm_prescribed": {
                            "objective": self.active_otm.get("objective", {}) if self.active_otm else {},
                            "constraints": reflect_constraints,
                            "procedure_id": (self.active_otm or {}).get("metadata", {}).get("procedure_id"),
                        },
                        "outcome": outcome,
                    },
                }
            ))
            logger.info(f"[ORCHESTRATOR] Episode {episode_id[:8]} recorded "
                        f"(resolved=False, reflection requested).")
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Failed to record episode: {e}")

    #  Procedure Queue
    async def _advance_procedure(self) -> bool:
        """Advance to the next executable step. Returns True if a step was activated, False if done."""
        proc = self.active_procedure
        if not proc or proc.state != ProcedureState.ACTIVE:
            return False

        now_str = datetime.now(timezone.utc).isoformat()

        while proc.current_step_idx + 1 < len(proc.steps):
            proc.current_step_idx += 1
            step = proc.steps[proc.current_step_idx]

            # Skip branch-only steps in normal linear flow
            if step.branch_only and not proc.branched_to:
                skip_msg = (f"[{now_str}] Procedure step {step.order} '{step.name}' SKIPPED: "
                            f"branch-only (not reached via conditional jump)")
                logger.info(f"[ORCHESTRATOR] {skip_msg}")
                self._append_to_active_log(skip_msg)
                await self._publish_procedure_update("step_skipped", step)
                continue

            # Clear branch flag only after consuming a non-branch-only step
            if not step.branch_only:
                proc.branched_to = False

            # Health check step: evaluate immediately against latest telemetry
            if step.health_check:
                passed, details = self._evaluate_health_check(step)
                event = "health_check_passed" if passed else "health_check_failed"
                await self._publish_procedure_update(event, step)
                if passed:
                    ok_msg = (f"[{now_str}] Health check '{step.name}' PASSED: {details}")
                    logger.info(f"[ORCHESTRATOR] {ok_msg}")
                    self._append_to_active_log(ok_msg)
                    proc.step_failures = 0  # Reset failures on success
                    continue  # Advance to next step
                else:
                    fail_msg = (f"[{now_str}] Health check '{step.name}' FAILED: {details}")
                    logger.warning(f"[ORCHESTRATOR] {fail_msg}")
                    self._append_to_active_log(fail_msg)
                    
                    # Try to branch if the user explicitly defined a mechanical fallback
                    if step.on_fail_goto:
                        target_idx = self._find_step_index_by_name(proc, step.on_fail_goto)
                        if target_idx is not None:
                            branch_msg = (f"[{now_str}] Branching to alternative step '{step.on_fail_goto}'")
                            logger.info(f"[ORCHESTRATOR] {branch_msg}")
                            self._append_to_active_log(branch_msg)
                            proc.current_step_idx = target_idx - 1  # -1 because loop increments
                            proc.branched_to = True
                            continue
                    
                    # If we have tried to adapt 3 times for this step, escalate to Cognitive Core for a scenario switch instead of just an OTM adaptation
                    proc.step_failures += 1
                    if proc.step_failures >= 3:
                        last_tried_scenario = proc.scenario_id
                        self.scenario_failure_count += 1
                        
                        failed_metric = step.health_check["checks"][0]["metric"]
                        val = self.last_kpi_snapshot.get("CellMetrics", {}).get(failed_metric, 0.0)
                        
                        # fallback and human escalation... i dont know if this is needed
                        if self.scenario_failure_count >= 2:
                            logger.critical(f"[ESCALATION] Multiple scenarios failed to resolve anomaly on '{failed_metric}'. Applying Safe Baseline and FREEZING.")
                            await self._publish_procedure_update("procedure_failed", step)
                            
                            # Wipe the broken procedure
                            self.active_procedure = None
                            self.scenario_failure_count = 0  # Reset so it can be clean when humans take over
                            
                            # Construct a safe, deterministic OTM with an Autopsy
                            baseline_otm = {
                                "version": "1.0",
                                "objective": {"service": "mbb", "kpi": "DRB_PdcpSduDelayDl", "aggregation": "mean", "unit": "ms", "maximize": False},
                                "constraints": [
                                    {"service": "mbb", "kpi": "dl_mcs_max", "operator": "le", "threshold": 20.0, "unit": "", "id": "SAFE_MCS"},
                                    {"service": "mbb", "kpi": "tx_power_dbm", "operator": "ge", "threshold": 46.0, "unit": "dBm", "id": "SAFE_PWR"}
                                ],
                                "metadata": {
                                    "timescale": "10s_window",
                                    "procedure_id": None,
                                    "is_critical_fallback": True,  # Flag for the Optimizer UI formatting
                                    "adaptation_log": [
                                        f"CRITICAL CRASH: AI exhausted all attempts to fix '{failed_metric}'.",
                                        f"Last attempted scenario '{last_tried_scenario}' aborted.",
                                        f"Fatal Metric: {failed_metric} degraded to {val:.2f}.",
                                        "Action taken: Reverted to deterministic safe mode (MCS <= 20, Tx Power >= 46).",
                                        "SYSTEM FROZEN: Autonomous control disabled. Awaiting manual human override."
                                    ]
                                }
                            }
                            
                            # Update our active OTM tracker
                            self.active_otm = baseline_otm
                            
                            # Publish the safe constraints directly to the execution layer
                            asyncio.create_task(self.bus.pub("intent.execute", make_msg("orch", "NEW_TARGET", "v1", self.active_otm)))
                            
                            # Lock the system. It will ignore new anomalies until timeout or human intervention.
                            self.state = IntentState.ESCALATED
                            self.escalated_at = time.time()
                            logger.critical("[ORCHESTRATOR] System locked in ESCALATED state. "
                                            f"Auto-unlock in {self.ESCALATED_TIMEOUT}s or via manual intent.")
                            
                            return True
                            
                        # SCENARIO SWITCH
                        logger.warning(f"[ORCHESTRATOR] Health check failed 3 times. Escalating '{failed_metric}' issue to Cognitive Core for Scenario Switch.")
                        await self._publish_procedure_update("procedure_failed", step)
                        await self._rollback_procedure()
                        
                        self.state = IntentState.THINKING
                        self.thinking_start_time = time.time()
                        
                        asyncio.create_task(self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "scenario_failed",
                            "failed_procedure": last_tried_scenario,  # Tell the LLM what not to use again
                            "metric": failed_metric,
                            "value": val
                        })))
                        return True
                        
                    logger.warning(f"[ORCHESTRATOR] Routing health check failure to AI Adaptation (attempt {proc.step_failures}/3).")
                    self.state = IntentState.THINKING
                    self.thinking_start_time = time.time()
                    self.e2_acks_received = 0
                    
                    # Step back so we re-evaluate this exact health check next loop
                    proc.current_step_idx -= 1
                    
                    # Extract the failing metric to give the LLM context
                    failed_metric = step.health_check["checks"][0]["metric"]
                    val = self.last_kpi_snapshot.get("CellMetrics", {}).get(failed_metric, 0.0)
                    
                    # Let LLM adapt the OTM if the health check failed.
                    asyncio.create_task(self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                        "type": "adapt_otm", 
                        "metric": failed_metric,
                        "value": val,
                        "previous_otm": self.active_otm
                    })))
                    return True  # Keep procedure active, wait for AI to generate new OTM

            # Skip non-capable steps without health checks
            if not step.ns3_capable or step.otm_fragment is None:
                skip_msg = (f"[{now_str}] Procedure step {step.order} '{step.name}' SKIPPED: "
                            f"{step.ns3_skip_reason or 'no OTM fragment'}")
                logger.info(f"[ORCHESTRATOR] {skip_msg}")
                self._append_to_active_log(skip_msg)
                await self._publish_procedure_update("step_skipped", step)
                continue

            # Found an actionable step: merge its constraints into active OTM
            self._merge_step_into_otm(step)
            step_msg = (f"[{now_str}] Procedure step {step.order} '{step.name}' ACTIVATED: "
                        f"{step.description}")
            logger.info(f"[ORCHESTRATOR] {step_msg}")
            self._append_to_active_log(step_msg)

            # Re-enter ASSURANCE for this step
            self.state = IntentState.ASSURANCE
            self.assurance_end_time = time.time() + 30.0
            self.e2_acks_received = 0

            # Publish updated OTM so constraint executor picks up the new constraints
            await self.bus.pub("intent.execute", make_msg(
                "opt", "NEW_TARGET", "v1", self.active_otm))
            await self._publish_procedure_update("step_activated", step)
            return True

        return False  # No more steps

    def _merge_step_into_otm(self, step):
        """Merge a procedure step's OTM fragment constraints into the active OTM."""
        if not self.active_otm or not step.otm_fragment:
            return

        existing = self.active_otm.setdefault("constraints", [])
        existing_ids = {c.get("id") for c in existing}

        for new_c in step.otm_fragment.get("constraints", []):
            cid = new_c.get("id")
            if cid in existing_ids:
                # Update existing constraint (same PROC_* id from a previous step)
                for ec in existing:
                    if ec.get("id") == cid:
                        ec["operator"] = new_c["operator"]
                        ec["threshold"] = new_c["threshold"]
                        ec["origin"] = "procedure_step"  # Tag for constraint executor
                        break
            else:
                tagged = dict(new_c)
                tagged["origin"] = "procedure_step"  # Tag for constraint executor
                existing.append(tagged)

    def _evaluate_health_check(self, step: ProcedureStep) -> tuple:
        """Evaluate health check conditions against a 10-second rolling telemetry window.

        Returns (passed: bool, details: str).
        """
        checks = step.health_check.get("checks", [])
        now = time.time()
        
        # Ensure we are only looking at data from the last 10 seconds
        valid_history = [raw for t, raw in self.kpi_history if now - t <= self.HEALTH_CHECK_WINDOW]
        
        if not valid_history:
            return False, "FAILED: No telemetry data available in the 10-second window."

        results = []
        all_passed = True
        
        for chk in checks:
            metric = chk["metric"]
            op = chk["operator"]
            threshold = float(chk["threshold"])

            # Collect all values for this metric over the 10s window
            window_values = []
            
            for raw in valid_history:
                cell = raw.get("CellMetrics", {})
                ues = raw.get("UEMetrics", [])
                
                val = cell.get(metric)
                
                # If not a cell metric, average across all UEs for this specific timestamp
                if val is None and ues:
                    ue_vals = []
                    for ue in ues:
                        if metric in ue and ue[metric] is not None:
                            try:
                                ue_vals.append(float(ue[metric]))
                            except (ValueError, TypeError):
                                pass
                    if ue_vals:
                        val = sum(ue_vals) / len(ue_vals)
                
                if val is not None:
                    window_values.append(float(val))

            # --- Evaluation ---
            if not window_values:
                # FIX applied here: NO DATA triggers a failure instead of passing
                results.append(f"{metric}: NO DATA (Connection Dropped?)")
                all_passed = False
                continue

            # Calculate the mean over the health check window
            avg_val = sum(window_values) / len(window_values)

            if op == "le":
                ok = avg_val <= threshold
            elif op == "lt":
                ok = avg_val < threshold
            elif op == "ge":
                ok = avg_val >= threshold
            elif op == "gt":
                ok = avg_val > threshold
            else:
                ok = False

            status = "OK" if ok else "FAIL"
            results.append(f"{metric}={avg_val:.2f}avg (N={len(window_values)}) {op} {threshold} -> {status}")
            
            if not ok:
                all_passed = False

        return all_passed, "; ".join(results) if results else "no checks defined"

    @staticmethod
    def _find_step_index_by_name(proc: ActiveProcedure, step_name: str):
        """Find the index of a procedure step by its name."""
        for i, s in enumerate(proc.steps):
            if s.name == step_name:
                return i
        logger.warning(f"[ORCHESTRATOR] Branch target step '{step_name}' not found in procedure.")
        return None

    async def _complete_procedure(self):
        """Mark the active procedure as completed."""
        proc = self.active_procedure
        if not proc:
            return
        proc.state = ProcedureState.COMPLETED
        now_str = datetime.now(timezone.utc).isoformat()
        msg = f"[{now_str}] Procedure '{proc.scenario_name}' COMPLETED ({proc.current_step_idx + 1}/{len(proc.steps)} steps processed)"
        logger.info(f"[ORCHESTRATOR] {msg}")
        self._append_to_active_log(msg)

        await self._publish_procedure_update("procedure_completed")

        self.active_procedure = None

    async def _rollback_procedure(self):
        """Rollback to pre-procedure OTM after repeated failures."""
        proc = self.active_procedure
        if not proc:
            return
        
        proc.state = ProcedureState.FAILED
        now_str = datetime.now(timezone.utc).isoformat()
        msg = f"[{now_str}] Procedure '{proc.scenario_name}' FAILED at step {proc.current_step_idx}. Rolling back."
        logger.warning(f"[ORCHESTRATOR] {msg}")
        self._append_to_active_log(msg)

        await self._publish_procedure_update("procedure_failed")

        if proc.pre_procedure_otm:
            self.active_otm = proc.pre_procedure_otm
            await self.bus.pub("intent.execute", make_msg("orch", "SCHEDULE_REVERT", "v1", self.active_otm))
        else:
            self.active_otm = None

        self.active_procedure = None
        self._clear_anomaly_state()
        self.state = IntentState.MONITORING

    def _append_to_active_log(self, entry: str):
        """Append an entry to the active OTM's adaptation_log."""
        if self.active_otm:
            self.active_otm.setdefault("metadata", {}).setdefault("adaptation_log", []).append(entry)

    async def _publish_procedure_update(self, event: str, step=None):
        """Publish a procedure status update for the dashboard."""
        proc = self.active_procedure
        if not proc:
            return
        payload = {
            "event": event,
            "scenario_id": proc.scenario_id,
            "scenario_name": proc.scenario_name,
            "current_step": proc.current_step_idx,
            "total_steps": len(proc.steps),
            "state": proc.state.name,
            "all_steps": [
                {"order": s.order, "name": s.name, "description": s.description} 
                for s in proc.steps
            ]
        }
        if step:
            payload["step_name"] = step.name
            payload["step_description"] = step.description
            payload["ns3_capable"] = step.ns3_capable
            payload["is_health_check"] = step.health_check is not None
            payload["is_branch_only"] = step.branch_only
            
        await self.bus.pub("procedure.update", make_msg("orch", "PROCEDURE", "v1", payload))

    #  Graceful Cancel & Schedule Broadcast

    async def _apply_cancel_revert(self):
        """Apply deferred revert from a gracefully cancelled scheduled intent."""
        revert_otm = self.pending_cancel_revert
        self.pending_cancel_revert = None
        now_str = datetime.now(timezone.utc).isoformat()

        if revert_otm:
            msg = f"[{now_str}] Graceful cancel revert applied. Restored pre-scheduled OTM."
            logger.info(f"[ORCHESTRATOR] {msg}")
            self._append_to_active_log(msg)
            self.active_otm = revert_otm
            await self.bus.pub("intent.execute", make_msg("orch", "SCHEDULE_REVERT", "v1", revert_otm))
        else:
            self.active_otm = None
            logger.info("[ORCHESTRATOR] Graceful cancel revert applied. No previous OTM to restore.")

        await self._broadcast_schedule_list()

    async def _broadcast_schedule_list(self):
        """Broadcast the full schedule queue state to the dashboard."""
        items = []
        for si in self.schedule_queue:
            items.append({
                "id": si.id,
                "raw_text": si.raw_text,
                "state": si.state.name,
                "activate_at": si.activate_at.isoformat(),
                "deactivate_at": si.deactivate_at.isoformat() if si.deactivate_at else None,
                "created_at": si.created_at.isoformat(),
                "cancel_pending": (si.state == ScheduleState.CANCELLED
                                   and self.pending_cancel_revert is not None),
            })
        await self.bus.pub("schedule.update", make_msg("orch", "SCHEDULE", "v1", {"items": items}))

    #  Temporal Scheduler

    async def _enqueue_scheduled_intent(self, otm: dict, corr_id: str) -> ScheduledIntent:
        """Create a ScheduledIntent from an OTM with temporal metadata and add it to the queue."""
        temporal = otm.get("metadata", {}).get("temporal_resolved", {})
        activate_at = datetime.fromisoformat(temporal["activate_at_utc"])
        deactivate_at_str = temporal.get("deactivate_at_utc")
        deactivate_at = datetime.fromisoformat(deactivate_at_str) if deactivate_at_str else None
        raw_text = otm.get("metadata", {}).get("original_text", "scheduled intent")

        si = ScheduledIntent(
            id=str(_uuid.uuid4()),
            raw_text=raw_text,
            otm=otm,
            activate_at=activate_at,
            deactivate_at=deactivate_at,
            corr_id=corr_id,
        )
        self.schedule_queue.append(si)
        self.schedule_queue.sort(key=lambda x: x.activate_at)

        logger.info(f"[ORCHESTRATOR] Scheduled intent '{si.id[:8]}' for activation at {activate_at.isoformat()}"
                    f"{f', expires at {deactivate_at.isoformat()}' if deactivate_at else ', no expiry'}")
        await self._broadcast_schedule_list()
        return si

    async def _cancel_scheduled_intent(self, schedule_id: str):
        """Cancel a scheduled intent. For ACTIVE intents, defer revert until current work completes."""
        for si in self.schedule_queue:
            if si.id != schedule_id:
                continue
            if si.state in (ScheduleState.EXPIRED, ScheduleState.CANCELLED):
                logger.info(f"[ORCHESTRATOR] Intent '{si.id[:8]}' already {si.state.name}. Ignoring cancel.")
                return

            prev_state = si.state
            si.state = ScheduleState.CANCELLED
            now_str = datetime.now(timezone.utc).isoformat()

            if prev_state == ScheduleState.ACTIVE:
                # Graceful cancel: defer revert until system returns to MONITORING
                self.pending_cancel_revert = si.pre_intent_otm
                cancel_msg = (f"[{now_str}] Scheduled intent '{si.raw_text}' CANCELLED. "
                              f"Graceful revert pending — current work will finish before reverting.")
                logger.info(f"[ORCHESTRATOR] {cancel_msg}")
                self._append_to_active_log(cancel_msg)
            else:
                logger.info(f"[ORCHESTRATOR] Cancelled PENDING intent '{si.id[:8]}': {si.raw_text}")

            await self._broadcast_schedule_list()
            return

        logger.warning(f"[ORCHESTRATOR] Cancel requested for unknown schedule_id '{schedule_id}'.")

    async def _tick_scheduler(self):
        """Check for pending activations and active expirations every loop tick."""
        now_utc = datetime.now(timezone.utc)
        schedule_changed = False

        # Activate pending intents whose time has come
        for si in self.schedule_queue:
            if si.state != ScheduleState.PENDING:
                continue
            if now_utc < si.activate_at:
                continue
            # Only activate when the state machine is idle
            if self.state != IntentState.MONITORING:
                logger.debug(f"[ORCHESTRATOR] Scheduler: deferring activation of '{si.id[:8]}' — state is {self.state.name}")
                continue

            logger.info(f"[ORCHESTRATOR] Activating scheduled intent '{si.id[:8]}': {si.raw_text}")
            self._clear_anomaly_state()  # Prevent stale anomaly data from contaminating episode recording
            # Auto-expire any existing ACTIVE scheduled intent and unwind its OTM
            for other in self.schedule_queue:
                if other.state == ScheduleState.ACTIVE:
                    other.state = ScheduleState.EXPIRED
                    self.active_otm = other.pre_intent_otm  # Revert before snapshotting
                    logger.info(f"[ORCHESTRATOR] Auto-expired '{other.id[:8]}' (superseded by '{si.id[:8]}')")
            si.pre_intent_otm = self.active_otm  # Now captures the correct baseline
            si.state = ScheduleState.ACTIVE
            self._apply_otm(si.otm)
            # Republish so constraint executor picks up the correct OTM at activation time
            await self.bus.pub("intent.execute", make_msg("orch", "SCHEDULE_ACTIVATION", "v1", si.otm))
            schedule_changed = True
            break  # one activation per tick to respect the state machine

        # Expire active intents whose window has closed
        for si in self.schedule_queue:
            if si.state != ScheduleState.ACTIVE:
                continue
            if si.deactivate_at is None or now_utc < si.deactivate_at:
                continue

            logger.info(f"[ORCHESTRATOR] Scheduled intent '{si.id[:8]}' EXPIRED. Auto-reverting.")
            si.state = ScheduleState.EXPIRED

            revert_otm = si.pre_intent_otm
            if revert_otm:
                logger.info("[ORCHESTRATOR] Reverting to pre-scheduled OTM.")
                await self.bus.pub("intent.execute", make_msg("orch", "SCHEDULE_REVERT", "v1", revert_otm))
                self.active_otm = revert_otm
            else:
                self.active_otm = None

            self.state = IntentState.MONITORING
            schedule_changed = True

        # Garbage-collect finished intents (keep last 20)
        active = [si for si in self.schedule_queue if si.state in (ScheduleState.PENDING, ScheduleState.ACTIVE)]
        finished = [si for si in self.schedule_queue if si.state in (ScheduleState.EXPIRED, ScheduleState.CANCELLED)]
        self.schedule_queue = active + finished[-20:]

        if schedule_changed:
            await self._broadcast_schedule_list()